"""
VQ-VAE for warehouse video tokenization.

Architecture
------------
Input  : (B, T, C, H, W)  — e.g. (4, 16, 3, 256, 256)
Encoder: 3-D convolution stack with 8× spatial + 4× temporal downsampling
         → (B, D, T//4, H//8, W//8)
Quantize: EMA-updated codebook  |  vocab 8192, dim 256
Decoder: 3-D deconvolution stack back to original resolution

All internal tensors use (B, C, T, H, W) layout (PyTorch conv3d convention).
The public API accepts (B, T, C, H, W) and converts internally.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: residual block for 3-D convolutions
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """Pre-norm residual block using 3-D grouped convolutions."""

    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, T, H, W)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class DownBlock3D(nn.Module):
    """Downsample spatially by 2 and (optionally) temporally by 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_down: bool = True,
        temporal_down: bool = False,
        groups: int = 32,
    ) -> None:
        super().__init__()
        stride_t = 2 if temporal_down else 1
        stride_s = 2 if spatial_down else 1
        stride = (stride_t, stride_s, stride_s)
        kernel = (3, 3, 3)
        padding = (1, 1, 1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class UpBlock3D(nn.Module):
    """Upsample spatially by 2 and (optionally) temporally by 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_up: bool = True,
        temporal_up: bool = False,
        groups: int = 32,
    ) -> None:
        super().__init__()
        scale_t = 2 if temporal_up else 1
        scale_s = 2 if spatial_up else 1
        self.scale = (scale_t, scale_s, scale_s)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    3-D convolutional encoder.

    Downsampling schedule (default):
      spatial : 2 × 2 × 2 = 8×
      temporal: 2 × 2     = 4×

    Output shape: (B, embedding_dim, T//4, H//8, W//8)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        channels = [base_channels * m for m in channel_mult]
        groups = 32

        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        ]

        # Stage 0: spatial down only
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[0], groups))
        layers.append(DownBlock3D(channels[0], channels[1], spatial_down=True, temporal_down=False))

        # Stage 1: spatial + temporal down
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[1], groups))
        layers.append(DownBlock3D(channels[1], channels[2], spatial_down=True, temporal_down=True))

        # Stage 2: spatial + temporal down
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[2], groups))
        layers.append(DownBlock3D(channels[2], channels[3], spatial_down=True, temporal_down=True))

        # Stage 3: bottleneck res blocks
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[3], groups))

        # Project to embedding_dim
        layers.append(nn.GroupNorm(groups, channels[3]))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Conv3d(channels[3], embedding_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, T, H, W) → (B, D, T', H', W')
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Mirror of Encoder — reconstructs (B, C_out, T, H, W) from latent."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()

        channels = [base_channels * m for m in channel_mult]
        groups = 32

        layers: list[nn.Module] = [
            nn.Conv3d(embedding_dim, channels[3], kernel_size=1),
        ]

        # Bottleneck
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[3], groups))

        # Stage 2 up
        layers.append(UpBlock3D(channels[3], channels[2], spatial_up=True, temporal_up=True))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[2], groups))

        # Stage 1 up
        layers.append(UpBlock3D(channels[2], channels[1], spatial_up=True, temporal_up=True))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[1], groups))

        # Stage 0 up
        layers.append(UpBlock3D(channels[1], channels[0], spatial_up=True, temporal_up=False))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[0], groups))

        # Output projection
        layers.append(nn.GroupNorm(groups, channels[0]))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:  # (B, D, T', H', W') → (B, C, T, H, W)
        return self.net(z)


# ---------------------------------------------------------------------------
# Vector Quantizer (EMA codebook updates + straight-through estimator)
# ---------------------------------------------------------------------------

class VQVAEOutput(NamedTuple):
    z_q: Tensor          # quantized latents  (B, D, T', H', W')
    loss: Tensor         # scalar commitment + codebook loss
    indices: Tensor      # code indices       (B, T', H', W')
    perplexity: Tensor   # codebook utilization proxy


class VectorQuantizer(nn.Module):
    """
    EMA-updated VQ layer.

    Uses exponential moving average to update codebook embeddings
    (no gradient through codebook), plus a commitment loss to pull
    encoder outputs toward the codebook.

    Reference: van den Oord et al., "Neural Discrete Representation Learning"
               (VQ-VAE, NeurIPS 2017)
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        # Codebook embedding table (not a gradient parameter — updated via EMA)
        embedding = torch.randn(codebook_size, embedding_dim)
        nn.init.uniform_(embedding, -1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("embedding", embedding)

        # EMA accumulators
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", embedding.clone())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distances(self, flat_z: Tensor) -> Tensor:
        """
        Compute squared L2 distance between each encoder output and all
        codebook entries.

        Args:
            flat_z: (N, D)
        Returns:
            distances: (N, K)
        """
        # ||z||^2 + ||e||^2 - 2 z·e^T
        z_sq = (flat_z ** 2).sum(dim=1, keepdim=True)          # (N, 1)
        e_sq = (self.embedding ** 2).sum(dim=1, keepdim=True).t()  # (1, K)
        cross = flat_z @ self.embedding.t()                     # (N, K)
        return z_sq + e_sq - 2.0 * cross

    def _ema_update(self, flat_z: Tensor, encoding_indices: Tensor) -> None:
        """Update codebook via EMA (only during training)."""
        # One-hot assignment matrix (N, K)
        one_hot = torch.zeros(
            flat_z.size(0), self.codebook_size,
            device=flat_z.device, dtype=flat_z.dtype
        )
        one_hot.scatter_(1, encoding_indices.unsqueeze(1), 1.0)

        # Cluster sizes
        cluster_size = one_hot.sum(dim=0)  # (K,)
        self.ema_cluster_size = (
            self.ema_decay * self.ema_cluster_size + (1 - self.ema_decay) * cluster_size
        )

        # Embedding sum
        embed_sum = one_hot.t() @ flat_z  # (K, D)
        self.ema_embedding_sum = (
            self.ema_decay * self.ema_embedding_sum + (1 - self.ema_decay) * embed_sum
        )

        # Normalise and write back
        n = self.ema_cluster_size.unsqueeze(1)  # (K, 1)
        updated = self.ema_embedding_sum / (n + self.ema_epsilon)
        self.embedding.copy_(updated)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> VQVAEOutput:
        """
        Args:
            z: encoder output  (B, D, T', H', W')
        Returns:
            VQVAEOutput
        """
        B, D, T, H, W = z.shape

        # Rearrange to (N, D)
        z_flat = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, D)  # (N, D)

        # Nearest codebook entry
        distances = self._distances(z_flat)
        indices = distances.argmin(dim=1)  # (N,)

        # EMA update (only during training)
        if self.training:
            self._ema_update(z_flat.detach(), indices.detach())

        # Quantized latents
        z_q_flat = self.embedding[indices]  # (N, D)
        z_q = z_q_flat.view(B, T, H, W, D).permute(0, 4, 1, 2, 3).contiguous()  # (B, D, T', H', W')

        # Straight-through estimator: copy gradients from z_q to z
        z_q_st = z + (z_q - z).detach()

        # Commitment loss only (codebook updated via EMA, not gradient)
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost

        # Perplexity (codebook utilization)
        avg_probs = (
            F.one_hot(indices, self.codebook_size).float().mean(dim=0)
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices_4d = indices.view(B, T, H, W)

        return VQVAEOutput(
            z_q=z_q_st,
            loss=commitment_loss,
            indices=indices_4d,
            perplexity=perplexity,
        )

    @torch.no_grad()
    def lookup(self, indices: Tensor) -> Tensor:
        """
        Decode indices → latent embeddings.

        Args:
            indices: (B, T', H', W') integer tensor
        Returns:
            z_q: (B, D, T', H', W')
        """
        B, T, H, W = indices.shape
        flat = indices.reshape(-1)
        z_q_flat = self.embedding[flat]  # (N, D)
        return z_q_flat.view(B, T, H, W, self.embedding_dim).permute(0, 4, 1, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# WarehouseVQVAE — the full model
# ---------------------------------------------------------------------------

class WarehouseVQVAEOutput(NamedTuple):
    recon: Tensor          # reconstructed video  (B, T, C, H, W)
    vq_loss: Tensor        # VQ commitment loss (scalar)
    recon_loss: Tensor     # reconstruction loss (scalar)
    indices: Tensor        # discrete token indices (B, T', H', W')
    perplexity: Tensor     # codebook perplexity


class WarehouseVQVAE(nn.Module):
    """
    Complete VQ-VAE for warehouse video clips.

    Input / output convention: (B, T, C, H, W)
    Internally uses (B, C, T, H, W) for nn.Conv3d.

    Default downsampling:
        spatial:  8× (H/8, W/8)
        temporal: 4× (T/4)

    With default H=W=256, T=16:
        latent shape per clip: (B, 256, 4, 32, 32)
        tokens per clip: 4 × 32 × 32 = 4096
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        codebook_size: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            embedding_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            embedding_dim=embedding_dim,
        )

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode video to discrete tokens.

        Args:
            x: (B, T, C, H, W)
        Returns:
            z_q:     quantized latent (B, D, T', H', W') — straight-through
            indices: token indices    (B, T', H', W')
            vq_loss: commitment loss scalar
        """
        # (B, T, C, H, W) → (B, C, T, H, W)
        x_bcthw = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encoder(x_bcthw)
        vq_out = self.quantizer(z)
        return vq_out.z_q, vq_out.indices, vq_out.loss

    def decode(self, z_q: Tensor) -> Tensor:
        """
        Decode quantized latents to video.

        Args:
            z_q: (B, D, T', H', W')
        Returns:
            recon: (B, T, C, H, W) in [0, 1] range after sigmoid
        """
        recon_bcthw = self.decoder(z_q)
        recon_bcthw = torch.sigmoid(recon_bcthw)
        # (B, C, T, H, W) → (B, T, C, H, W)
        return recon_bcthw.permute(0, 2, 1, 3, 4).contiguous()

    def decode_indices(self, indices: Tensor) -> Tensor:
        """
        Decode discrete token indices directly to video frames.

        Args:
            indices: (B, T', H', W') integer tensor
        Returns:
            recon: (B, T, C, H, W)
        """
        z_q = self.quantizer.lookup(indices)
        return self.decode(z_q)

    def forward(self, x: Tensor) -> WarehouseVQVAEOutput:
        """
        Full VQ-VAE forward pass.

        Args:
            x: (B, T, C, H, W) float32 in [0, 1]
        Returns:
            WarehouseVQVAEOutput
        """
        z_q, indices, vq_loss = self.encode(x)
        recon = self.decode(z_q)

        # Reconstruction loss (L1 + L2 blend)
        recon_loss = F.l1_loss(recon, x) + 0.1 * F.mse_loss(recon, x)

        # Perplexity from the quantizer's last forward call
        # Re-run quantizer to retrieve perplexity (already computed internally)
        x_bcthw = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encoder(x_bcthw)
        vq_out = self.quantizer(z)

        return WarehouseVQVAEOutput(
            recon=recon,
            vq_loss=vq_loss,
            recon_loss=recon_loss,
            indices=indices,
            perplexity=vq_out.perplexity,
        )

    def total_loss(self, output: WarehouseVQVAEOutput, recon_weight: float = 1.0) -> Tensor:
        """Scalar total loss for backward()."""
        return recon_weight * output.recon_loss + output.vq_loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"WarehouseVQVAE("
            f"codebook_size={self.codebook_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"params={self.num_parameters / 1e6:.1f}M)"
        )
