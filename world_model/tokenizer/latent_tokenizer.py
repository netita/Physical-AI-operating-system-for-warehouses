"""
Latent Tokenizer — KL-regularized VAE for continuous latent spaces.

Unlike the VQ-VAE this model produces a continuous Gaussian latent
distribution (mu, log_var) and uses the reparameterisation trick for
training.  At inference time the mean is used directly, avoiding the
quantisation rounding error and enabling smooth interpolation.

Architecture
------------
Input  : (B, T, C, H, W)
Encoder → mu (B, latent_dim, T', H', W')
        → log_var (same shape)
Decoder ← z sampled from N(mu, exp(0.5 * log_var))
Output : (B, T, C, H, W) reconstructed frames

Spatial downsampling : 8× (3 stages, each 2×)
Temporal downsampling: 4× (2 stages, each 2×)
Latent channels      : 16 (compact, suitable for transformer input)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warehousegpt.world_model.tokenizer.vqvae import (
    ResBlock3D,
    DownBlock3D,
    UpBlock3D,
)


# ---------------------------------------------------------------------------
# Encoder for the VAE (outputs mean and log-variance)
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    3-D convolutional encoder that outputs a Gaussian latent distribution.

    Output: (mu, log_var), each of shape (B, latent_dim, T//4, H//8, W//8)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        channels = [base_channels * m for m in channel_mult]
        groups = 32

        # Shared backbone (same as VQ-VAE encoder without the final projection)
        backbone: list[nn.Module] = [
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        ]

        for _ in range(num_res_blocks):
            backbone.append(ResBlock3D(channels[0], groups))
        backbone.append(DownBlock3D(channels[0], channels[1], spatial_down=True, temporal_down=False))

        for _ in range(num_res_blocks):
            backbone.append(ResBlock3D(channels[1], groups))
        backbone.append(DownBlock3D(channels[1], channels[2], spatial_down=True, temporal_down=True))

        for _ in range(num_res_blocks):
            backbone.append(ResBlock3D(channels[2], groups))
        backbone.append(DownBlock3D(channels[2], channels[3], spatial_down=True, temporal_down=True))

        for _ in range(num_res_blocks):
            backbone.append(ResBlock3D(channels[3], groups))

        backbone.append(nn.GroupNorm(groups, channels[3]))
        backbone.append(nn.SiLU(inplace=True))

        self.backbone = nn.Sequential(*backbone)

        # Dual heads: mean and log-variance
        self.mu_head = nn.Conv3d(channels[3], latent_dim, kernel_size=1)
        self.log_var_head = nn.Conv3d(channels[3], latent_dim, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            mu:      (B, latent_dim, T', H', W')
            log_var: (B, latent_dim, T', H', W')
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-30.0, max=20.0)
        return mu, log_var


# ---------------------------------------------------------------------------
# Decoder for the VAE
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    """Mirror of VAEEncoder — reconstructs video from latent sample."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        groups = 32

        layers: list[nn.Module] = [
            nn.Conv3d(latent_dim, channels[3], kernel_size=1),
        ]

        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[3], groups))

        layers.append(UpBlock3D(channels[3], channels[2], spatial_up=True, temporal_up=True))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[2], groups))

        layers.append(UpBlock3D(channels[2], channels[1], spatial_up=True, temporal_up=True))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[1], groups))

        layers.append(UpBlock3D(channels[1], channels[0], spatial_up=True, temporal_up=False))
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(channels[0], groups))

        layers.append(nn.GroupNorm(groups, channels[0]))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:  # (B, D, T', H', W') → (B, C, T, H, W)
        return self.net(z)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class LatentDistribution(NamedTuple):
    """Output of encode_video() — lazy sampling."""
    mu: Tensor       # (B, latent_dim, T', H', W')
    log_var: Tensor  # (B, latent_dim, T', H', W')

    def sample(self, deterministic: bool = False) -> Tensor:
        """Draw a latent sample (or return mean for deterministic decoding)."""
        if deterministic:
            return self.mu
        return LatentTokenizer.reparameterize_static(self.mu, self.log_var)

    def kl_loss(self) -> Tensor:
        """KL divergence from N(mu, sigma) to N(0, 1), averaged over batch."""
        return -0.5 * torch.mean(
            1.0 + self.log_var - self.mu.pow(2) - self.log_var.exp()
        )


class LatentTokenizerOutput(NamedTuple):
    recon: Tensor           # (B, T, C, H, W) reconstructed frames
    mu: Tensor              # (B, latent_dim, T', H', W')
    log_var: Tensor         # (B, latent_dim, T', H', W')
    recon_loss: Tensor      # scalar
    kl_loss: Tensor         # scalar (KL divergence)
    total_loss: Tensor      # scalar (recon + beta * kl)


# ---------------------------------------------------------------------------
# LatentTokenizer
# ---------------------------------------------------------------------------

class LatentTokenizer(nn.Module):
    """
    KL-regularized VAE tokenizer for warehouse video.

    Produces a *continuous* latent space instead of discrete tokens.
    At training time samples from the posterior via reparameterization.
    At inference time uses the posterior mean (deterministic).

    Args:
        in_channels:    number of input image channels (default: 3 for RGB)
        base_channels:  base channel width before multipliers
        channel_mult:   channel width multipliers per stage
        num_res_blocks: residual blocks per stage
        latent_dim:     latent channel dimension (default: 16)
        beta:           KL weight (beta-VAE coefficient, default: 0.001)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        latent_dim: int = 16,
        beta: float = 0.001,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = VAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
        )
        self.decoder = VAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init log_var head to start near unit Gaussian
        nn.init.zeros_(self.encoder.log_var_head.weight)
        if self.encoder.log_var_head.bias is not None:
            nn.init.zeros_(self.encoder.log_var_head.bias)

    # ------------------------------------------------------------------
    # Reparameterization trick
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize_static(mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Sample z ~ N(mu, sigma^2) using the reparameterisation trick.

        z = mu + sigma * epsilon,  epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Instance method wrapper (useful for subclassing / mocking)."""
        if self.training:
            return self.reparameterize_static(mu, log_var)
        return mu  # deterministic at eval time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_video(self, x: Tensor) -> LatentDistribution:
        """
        Encode a video clip to a Gaussian latent distribution.

        Args:
            x: (B, T, C, H, W) float32 in [0, 1]
        Returns:
            LatentDistribution with .mu and .log_var
        """
        # (B, T, C, H, W) → (B, C, T, H, W)
        x_bcthw = x.permute(0, 2, 1, 3, 4).contiguous()
        mu, log_var = self.encoder(x_bcthw)
        return LatentDistribution(mu=mu, log_var=log_var)

    def decode_latents(self, z: Tensor) -> Tensor:
        """
        Decode latent sample to reconstructed frames.

        Args:
            z: (B, latent_dim, T', H', W')
        Returns:
            recon: (B, T, C, H, W) in [0, 1]
        """
        recon_bcthw = self.decoder(z)
        recon_bcthw = torch.sigmoid(recon_bcthw)
        # (B, C, T, H, W) → (B, T, C, H, W)
        return recon_bcthw.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: Tensor) -> LatentTokenizerOutput:
        """
        Full VAE forward pass with loss computation.

        Args:
            x: (B, T, C, H, W) float32 in [0, 1]
        Returns:
            LatentTokenizerOutput
        """
        dist = self.encode_video(x)
        z = self.reparameterize(dist.mu, dist.log_var)
        recon = self.decode_latents(z)

        # Losses
        recon_loss = F.l1_loss(recon, x) + 0.1 * F.mse_loss(recon, x)
        kl_loss = dist.kl_loss()
        total_loss = recon_loss + self.beta * kl_loss

        return LatentTokenizerOutput(
            recon=recon,
            mu=dist.mu,
            log_var=dist.log_var,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def encode_deterministic(self, x: Tensor) -> Tensor:
        """
        Encode video to posterior mean (no sampling).
        Suitable for inference / downstream transformer input.

        Args:
            x: (B, T, C, H, W)
        Returns:
            mu: (B, latent_dim, T', H', W')
        """
        return self.encode_video(x).mu

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Encode and decode a video clip deterministically.

        Args:
            x: (B, T, C, H, W)
        Returns:
            recon: (B, T, C, H, W)
        """
        mu = self.encode_deterministic(x)
        return self.decode_latents(mu)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"LatentTokenizer("
            f"latent_dim={self.latent_dim}, "
            f"beta={self.beta}, "
            f"params={self.num_parameters / 1e6:.1f}M)"
        )
