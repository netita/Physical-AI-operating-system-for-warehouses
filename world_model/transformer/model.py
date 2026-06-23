"""
WarehouseWorldModel — Causal Video Transformer.

Architecture
------------
1. Spatial patch embedding (16×16 patches per frame)
2. Temporal position encoding (RoPE applied in attention)
3. N × TransformerBlock (causal self-attention + action cross-attention + FFN)
4. Output heads:
     a) next-frame token logits  (B, L, codebook_size)
     b) risk score               (B, 1)
     c) occupancy grid logits    (B, T', H', W', 3)  [free / occupied / unknown]

Default configuration targets ~1B parameters:
    d_model=2048, num_layers=24, num_heads=16, ffn_multiplier=4
    (actual parameter count depends on codebook_size and head sizes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warehousegpt.world_model.transformer.attention import TransformerBlock


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class WarehouseTransformerConfig:
    """
    Hyper-parameters for the WarehouseWorldModel.

    1B parameter default configuration
    ------------------------------------
    d_model=2048, num_layers=24, num_heads=16
    → ~2048*24*(12*2048 + 2*2048*2730) ≈ 1.1B params

    Smaller configs for development / ablation
    ------------------------------------------
    300M: d_model=1536, num_layers=18, num_heads=12
    100M: d_model=1024, num_layers=12, num_heads=8
     50M: d_model=768,  num_layers=8,  num_heads=8
    """

    # --- Token space ---
    codebook_size: int = 8192
    """VQ-VAE vocabulary size (number of discrete video tokens)."""

    # --- Frame / patch geometry ---
    image_height: int = 256
    """Input frame height (pixels)."""
    image_width: int = 256
    """Input frame width (pixels)."""
    patch_size: int = 16
    """Spatial patch size (pixels per side)."""

    @property
    def num_patches_h(self) -> int:
        return self.image_height // self.patch_size

    @property
    def num_patches_w(self) -> int:
        return self.image_width // self.patch_size

    @property
    def tokens_per_frame(self) -> int:
        return self.num_patches_h * self.num_patches_w

    # --- Sequence ---
    max_frames: int = 64
    """Maximum number of frames in a context window."""

    @property
    def max_seq_len(self) -> int:
        return self.max_frames * self.tokens_per_frame

    # --- Transformer ---
    d_model: int = 2048
    """Hidden dimension of the transformer."""
    num_layers: int = 24
    """Number of transformer blocks."""
    num_heads: int = 16
    """Number of attention heads."""
    ffn_multiplier: float = 8.0 / 3.0
    """Feed-forward dimension multiplier (SwiGLU default)."""
    dropout: float = 0.0
    """Dropout rate (0 for inference-mode training with large models)."""

    # --- Action conditioning ---
    action_dim: int = 6
    """
    Raw action vector dimension.
    Default components:
        [vx, vy, omega, fork_height, fork_tilt, load_weight]
    """
    action_embed_dim: int = 256
    """Projected action embedding dimension for cross-attention."""
    action_seq_len: int = 8
    """Number of action context tokens (expand single action → sequence)."""
    context_dim: Optional[int] = None
    """
    Cross-attention context dim. Set automatically to action_embed_dim
    if None (done in __post_init__).
    """

    # --- Output heads ---
    occupancy_classes: int = 3
    """Number of occupancy classes: 0=free, 1=occupied, 2=unknown."""

    # --- Initialisation ---
    init_std: float = 0.02
    """Standard deviation for weight initialisation."""

    # --- Tokenizer integration ---
    vqvae_spatial_downsample: int = 8
    """Spatial downsampling factor of the paired VQ-VAE."""
    vqvae_temporal_downsample: int = 4
    """Temporal downsampling factor of the paired VQ-VAE."""

    def __post_init__(self) -> None:
        if self.context_dim is None:
            self.context_dim = self.action_embed_dim

    @property
    def ffn_dim(self) -> int:
        raw = int(self.d_model * self.ffn_multiplier)
        return ((raw + 63) // 64) * 64  # round to multiple of 64


# ---------------------------------------------------------------------------
# Spatial patch embedding
# ---------------------------------------------------------------------------

class SpatialPatchEmbedding(nn.Module):
    """
    Embeds a sequence of video frames as spatial patch tokens.

    Each frame is divided into non-overlapping (patch_size × patch_size) patches.
    Each patch is linearly projected to d_model.

    Supports two input modes:
      - Pixel frames:   (B, T, C, H, W)   → raw RGB / depth frames
      - Token frames:   (B, T, Ht, Wt)    → discrete VQ-VAE token indices

    In token mode the embedding table replaces the linear patch projection.
    """

    def __init__(self, cfg: WarehouseTransformerConfig, in_channels: int = 3) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size

        patch_dim = in_channels * cfg.patch_size * cfg.patch_size

        # For pixel-space input: 2D conv as patch extractor + projection
        self.pixel_proj = nn.Conv2d(
            in_channels,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

        # For discrete token input: codebook embedding table
        self.token_embed = nn.Embedding(cfg.codebook_size, cfg.d_model)

        # Learned spatial position bias (per-patch, shared across frames)
        num_spatial = cfg.num_patches_h * cfg.num_patches_w
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, num_spatial, cfg.d_model))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=cfg.init_std)

    def embed_pixels(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            tokens: (B, T * num_patches, d_model)
        """
        B, T, C, H, W = x.shape
        # Process each frame independently
        x_2d = x.reshape(B * T, C, H, W)
        patches = self.pixel_proj(x_2d)  # (B*T, d_model, Ht, Wt)
        patches = patches.flatten(2).transpose(1, 2)  # (B*T, Ht*Wt, d_model)
        patches = patches.view(B, T, -1, self.cfg.d_model)  # (B, T, Ht*Wt, d_model)
        patches = patches + self.spatial_pos_embed  # broadcast over B, T
        return patches.reshape(B, T * patches.shape[2], self.cfg.d_model)

    def embed_tokens(self, idx: Tensor) -> Tensor:
        """
        Args:
            idx: (B, T, Ht, Wt) integer VQ-VAE token indices
        Returns:
            tokens: (B, T * Ht * Wt, d_model)
        """
        B, T, Ht, Wt = idx.shape
        flat_idx = idx.reshape(B, T, Ht * Wt)
        emb = self.token_embed(flat_idx)  # (B, T, Ht*Wt, d_model)
        emb = emb + self.spatial_pos_embed  # (B, T, Ht*Wt, d_model)
        return emb.reshape(B, T * Ht * Wt, self.cfg.d_model)


# ---------------------------------------------------------------------------
# Action encoder (projects raw actions → cross-attention context)
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """
    Encodes forklift action vectors to a sequence of context tokens.

    Input : (B, T, action_dim) — one action per frame
    Output: (B, T * action_seq_len, action_embed_dim) — expanded context

    Uses a 2-layer MLP + optional GRU for temporal correlation.
    """

    def __init__(self, cfg: WarehouseTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.action_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.action_embed_dim, cfg.action_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.action_embed_dim, cfg.action_embed_dim * cfg.action_seq_len),
        )

    def forward(self, actions: Tensor) -> Tensor:
        """
        Args:
            actions: (B, T, action_dim)
        Returns:
            context: (B, T * action_seq_len, action_embed_dim)
        """
        B, T, _ = actions.shape
        projected = self.mlp(actions)  # (B, T, action_embed_dim * action_seq_len)
        projected = projected.view(
            B, T * self.cfg.action_seq_len, self.cfg.action_embed_dim
        )
        return projected


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class NextTokenHead(nn.Module):
    """Projects transformer output to next-frame token logits."""

    def __init__(self, cfg: WarehouseTransformerConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.codebook_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            logits: (B, L, codebook_size)
        """
        return self.proj(self.norm(x))


class RiskScoreHead(nn.Module):
    """
    Predicts a scalar risk score in [0, 1] from the mean-pooled
    transformer representation.

    Risk captures: collision probability, fire hazard, congestion.
    """

    def __init__(self, cfg: WarehouseTransformerConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 4),
            nn.GELU(),
            nn.Linear(cfg.d_model // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            risk: (B, 1) in [0, 1]
        """
        pooled = self.norm(x).mean(dim=1)  # (B, d_model)
        return torch.sigmoid(self.mlp(pooled))


class OccupancyHead(nn.Module):
    """
    Predicts per-cell occupancy logits from token-level features.

    Reshapes token sequence back to (B, T', Ht, Wt, d_model)
    then projects to occupancy_classes.
    """

    def __init__(self, cfg: WarehouseTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.occupancy_classes)

    def forward(self, x: Tensor, T: int) -> Tensor:
        """
        Args:
            x: (B, L, d_model) where L = T * Ht * Wt
            T: number of time steps
        Returns:
            logits: (B, T, Ht, Wt, occupancy_classes)
        """
        B, L, D = x.shape
        Ht = self.cfg.num_patches_h
        Wt = self.cfg.num_patches_w
        x = self.norm(x)
        # Reshape to spatial grid
        x = x.view(B, T, Ht, Wt, D)
        return self.proj(x)  # (B, T, Ht, Wt, occupancy_classes)


# ---------------------------------------------------------------------------
# WarehouseWorldModel
# ---------------------------------------------------------------------------

class WorldModelOutput(NamedTuple):
    """Output container for WarehouseWorldModel.forward()."""
    next_token_logits: Tensor     # (B, L, codebook_size)
    risk_score: Tensor            # (B, 1)
    occupancy_logits: Tensor      # (B, T, Ht, Wt, occupancy_classes)
    hidden_states: Tensor         # (B, L, d_model) — for downstream heads


class WarehouseWorldModel(nn.Module):
    """
    Causal video transformer for warehouse world modeling.

    Supports two input modes (controlled by `input_mode` in forward()):
      "tokens"  — discrete VQ-VAE indices    (B, T, Ht, Wt)
      "pixels"  — raw video frames           (B, T, C, H, W)

    Action conditioning:
      actions   — (B, T, action_dim) forklift state vectors

    Default config produces ~1.1B parameters.
    """

    def __init__(
        self,
        cfg: Optional[WarehouseTransformerConfig] = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.cfg = cfg or WarehouseTransformerConfig()

        # Patch / token embedding
        self.patch_embed = SpatialPatchEmbedding(self.cfg, in_channels=in_channels)

        # Action encoder
        self.action_encoder = ActionEncoder(self.cfg)

        # Learned temporal position embedding
        # (additive, per frame, shared across spatial positions)
        self.temporal_pos_embed = nn.Embedding(self.cfg.max_frames, self.cfg.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.cfg.d_model,
                num_heads=self.cfg.num_heads,
                context_dim=self.cfg.context_dim,
                ffn_dim=self.cfg.ffn_dim,
                dropout=self.cfg.dropout,
                use_rope=True,
                max_seq_len=self.cfg.max_seq_len,
            )
            for _ in range(self.cfg.num_layers)
        ])

        # Output heads
        self.next_token_head = NextTokenHead(self.cfg)
        self.risk_head = RiskScoreHead(self.cfg)
        self.occupancy_head = OccupancyHead(self.cfg)

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        depth_scale = 1.0 / math.sqrt(2 * self.cfg.num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual projections (GPT-2 style)
        for block in self.blocks:
            nn.init.trunc_normal_(
                block.self_attn.out_proj.weight, std=std * depth_scale
            )
            nn.init.trunc_normal_(
                block.ffn.w2.weight, std=std * depth_scale
            )

    # ------------------------------------------------------------------
    # Temporal position encoding helper
    # ------------------------------------------------------------------

    def _add_temporal_pos(self, tokens: Tensor, T: int) -> Tensor:
        """
        Add per-frame temporal position embedding to the token sequence.

        Args:
            tokens: (B, T * S, d_model)  where S = spatial tokens per frame
            T:      number of frames
        Returns:
            tokens: same shape with temporal position added
        """
        B, L, D = tokens.shape
        S = L // T  # spatial tokens per frame

        # Create frame indices: [0,0,..., 1,1,..., T-1,T-1,...]
        frame_idx = torch.arange(T, device=tokens.device)  # (T,)
        frame_idx = frame_idx.unsqueeze(-1).expand(T, S)   # (T, S)
        frame_idx = frame_idx.reshape(T * S)               # (L,)

        pos_emb = self.temporal_pos_embed(frame_idx)  # (L, D)
        return tokens + pos_emb.unsqueeze(0)           # (B, L, D)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        video: Tensor,
        actions: Optional[Tensor] = None,
        input_mode: str = "tokens",
        kv_caches: Optional[list[Optional[dict[str, Tensor]]]] = None,
        seq_offset: int = 0,
    ) -> WorldModelOutput:
        """
        Args:
            video:       token mode:  (B, T, Ht, Wt) int64
                         pixel mode:  (B, T, C, H, W) float32
            actions:     (B, T, action_dim) float32  (optional)
            input_mode:  "tokens" | "pixels"
            kv_caches:   list of per-layer KV cache dicts (for generation)
            seq_offset:  position offset for KV-cache decoding
        Returns:
            WorldModelOutput
        """
        if input_mode == "tokens":
            T = video.shape[1]
            tokens = self.patch_embed.embed_tokens(video)   # (B, L, d_model)
        elif input_mode == "pixels":
            T = video.shape[1]
            tokens = self.patch_embed.embed_pixels(video)   # (B, L, d_model)
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r}")

        # Temporal position encoding
        tokens = self._add_temporal_pos(tokens, T)

        # Action conditioning context
        context: Optional[Tensor] = None
        if actions is not None:
            context = self.action_encoder(actions)  # (B, T*action_seq_len, action_embed_dim)

        # Initialise KV caches if not provided
        if kv_caches is None:
            kv_caches = [None] * self.cfg.num_layers

        # Transformer forward
        x = tokens
        new_caches: list[Optional[dict[str, Tensor]]] = []
        for block, cache in zip(self.blocks, kv_caches):
            x, updated_cache = block(x, context=context, kv_cache=cache, seq_offset=seq_offset)
            new_caches.append(updated_cache)

        # Output heads
        next_token_logits = self.next_token_head(x)       # (B, L, codebook_size)
        risk_score = self.risk_head(x)                    # (B, 1)
        occupancy_logits = self.occupancy_head(x, T)      # (B, T, Ht, Wt, classes)

        return WorldModelOutput(
            next_token_logits=next_token_logits,
            risk_score=risk_score,
            occupancy_logits=occupancy_logits,
            hidden_states=x,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"WarehouseWorldModel("
            f"d_model={cfg.d_model}, "
            f"num_layers={cfg.num_layers}, "
            f"num_heads={cfg.num_heads}, "
            f"codebook_size={cfg.codebook_size}, "
            f"params={self.num_parameters / 1e9:.2f}B)"
        )
