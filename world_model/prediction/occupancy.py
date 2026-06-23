"""
OccupancyForecaster — predicts binary/ternary occupancy grids from
transformer hidden states.

The forecaster attaches a lightweight UNet-style decoder head on top
of the world model's token features.  It upsamples from the latent
grid (Ht × Wt) back to the original frame resolution (H × W) and
predicts three occupancy classes per cell:
    0 = free
    1 = occupied (by worker, forklift, rack, etc.)
    2 = unknown  (outside camera frustum / occluded)

Input:  hidden_states (B, T, Ht, Wt, d_model)  from WarehouseWorldModel
Output: (B, T, H, W, 3)  per-pixel class logits

Architecture
------------
  features (B*T, d_model, Ht, Wt)
    │
    ├─ Skip connection encoded at each resolution
    ▼
  ConvBlock (d_model → 256)
    ▼
  UNet Up × 3  (256→128→64→32, each 2×)
    ▼
  Final conv → (B*T, 3, H, W)
    ▼
  Reshape → (B, T, H, W, 3)
"""

from __future__ import annotations

from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

def conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetEncoderBlock(nn.Module):
    """2 conv layers + 2× max pool for skip connections."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.conv(x)
        # Guard: only pool if spatial dims are large enough
        if skip.shape[2] > 1 and skip.shape[3] > 1:
            down = self.pool(skip)
        else:
            down = skip  # no-op — spatial size already 1×1
        return down, skip  # (down, skip)


class UNetDecoderBlock(nn.Module):
    """Bilinear upsample + concat skip + 2 conv layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch + skip_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Handle spatial size mismatch (odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# OccupancyForecaster
# ---------------------------------------------------------------------------

class OccupancyOutput(NamedTuple):
    logits: Tensor        # (B, T, H, W, num_classes)
    probs: Tensor         # (B, T, H, W, num_classes)  softmax
    grid: Tensor          # (B, T, H, W)              argmax class


class OccupancyForecaster(nn.Module):
    """
    UNet-style occupancy forecaster.

    Takes transformer hidden states and produces per-pixel occupancy maps.

    Args:
        d_model:         transformer hidden dimension (input channel count)
        num_classes:     occupancy classes (default: 3 — free/occupied/unknown)
        unet_channels:   channel widths for UNet encoder stages
        output_height:   target output height (must be 2^n * Ht)
        output_width:    target output width
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_classes: int = 3,
        unet_channels: tuple[int, ...] = (256, 128, 64, 32),
        output_height: int = 256,
        output_width: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.output_height = output_height
        self.output_width = output_width

        ch = unet_channels  # (C0, C1, C2, C3)

        # Input projection: d_model → C0 channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(d_model, ch[0], kernel_size=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder (downsampling) with skip connections
        self.enc1 = UNetEncoderBlock(ch[0], ch[0])      # Ht/2
        self.enc2 = UNetEncoderBlock(ch[0], ch[1])      # Ht/4
        self.enc3 = UNetEncoderBlock(ch[1], ch[2])      # Ht/8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            conv_bn_relu(ch[2], ch[2] * 2),
            conv_bn_relu(ch[2] * 2, ch[2]),
        )

        # Decoder (upsampling) with skip connections
        self.dec3 = UNetDecoderBlock(ch[2], ch[2], ch[2])
        self.dec2 = UNetDecoderBlock(ch[2], ch[1], ch[1])
        self.dec1 = UNetDecoderBlock(ch[1], ch[0], ch[0])

        # Final upsampling to original image size and classification
        self.final_up = nn.Sequential(
            nn.Upsample(size=(output_height, output_width), mode="bilinear", align_corners=False),
            conv_bn_relu(ch[0], ch[3]),
            nn.Conv2d(ch[3], num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor, T: int) -> OccupancyOutput:
        """
        Args:
            hidden_states: (B, L, d_model)  where L = T * Ht * Wt
            T:             number of time steps
        Returns:
            OccupancyOutput
        """
        B, L, D = hidden_states.shape
        S = L // T
        Ht = Wt = int(S ** 0.5)  # assume square spatial grid

        # Reshape: (B*T, d_model, Ht, Wt)
        x = hidden_states.view(B * T, Ht, Wt, D).permute(0, 3, 1, 2).contiguous()

        # Project to feature channels
        x = self.input_proj(x)  # (B*T, C0, Ht, Wt)

        # Encoder
        x1, skip1 = self.enc1(x)    # x1: (B*T, C0, Ht/2, Wt/2)
        x2, skip2 = self.enc2(x1)   # x2: (B*T, C1, Ht/4, Wt/4)
        x3, skip3 = self.enc3(x2)   # x3: (B*T, C2, Ht/8, Wt/8)

        # Bottleneck
        xb = self.bottleneck(x3)    # (B*T, C2, Ht/8, Wt/8)

        # Decoder
        xd3 = self.dec3(xb, skip3)  # (B*T, C2, Ht/4, Wt/4)
        xd2 = self.dec2(xd3, skip2) # (B*T, C1, Ht/2, Wt/2)
        xd1 = self.dec1(xd2, skip1) # (B*T, C0, Ht, Wt)

        # Final upsampling + classification
        logits_2d = self.final_up(xd1)  # (B*T, num_classes, H, W)
        H, W = logits_2d.shape[2:]

        # Reshape: (B, T, H, W, num_classes)
        logits = logits_2d.view(B, T, self.num_classes, H, W)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()

        probs = F.softmax(logits, dim=-1)
        grid = probs.argmax(dim=-1)  # (B, T, H, W)

        return OccupancyOutput(logits=logits, probs=probs, grid=grid)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_occupancy_grid(
        self,
        hidden_states: Tensor,
        T: int,
        horizon: Optional[int] = None,
    ) -> Tensor:
        """
        Predict binary occupancy grids (free / occupied / unknown).

        Args:
            hidden_states: (B, L, d_model)
            T:             total number of time steps in hidden_states
            horizon:       if set, only return the last `horizon` time steps
        Returns:
            grid: (B, T_out, H, W) int64 with values in {0, 1, 2}
        """
        out = self.forward(hidden_states, T)
        grid = out.grid  # (B, T, H, W)
        if horizon is not None:
            grid = grid[:, -horizon:]
        return grid

    def compute_loss(
        self,
        hidden_states: Tensor,
        target_grids: Tensor,
        T: int,
        class_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross-entropy loss against ground-truth occupancy labels.

        Args:
            hidden_states: (B, L, d_model)
            target_grids:  (B, T, H, W) int64 class labels
            T:             number of time steps
            class_weights: (num_classes,) optional per-class weighting
        Returns:
            loss: scalar
        """
        out = self.forward(hidden_states, T)
        B, T_out, H, W, C = out.logits.shape

        # Flatten for cross-entropy: (B*T*H*W, C) vs (B*T*H*W,)
        logits_flat = out.logits.view(-1, C)
        target_flat = target_grids.view(-1)

        return F.cross_entropy(
            logits_flat,
            target_flat,
            weight=class_weights,
            ignore_index=-1,
        )
