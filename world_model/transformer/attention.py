"""
Attention modules for the WarehouseGPT causal video transformer.

Modules
-------
RotaryPositionEmbedding   — RoPE for temporal + spatial positions
MultiHeadSelfAttention    — causal self-attention (FlashAttention2 via sdpa)
MultiHeadCrossAttention   — cross-attention for action conditioning
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes absolute position into the query and key vectors by rotating
    pairs of dimensions in the head space.

    Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary
               Position Embedding" (2021).

    Args:
        head_dim:   dimension of each attention head (must be even)
        max_seq_len: maximum sequence length to pre-compute (default: 8192)
        base:       RoPE base frequency (default: 10000)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies: shape (head_dim // 2,)
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache: (max_seq_len, head_dim)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate every pair of dimensions by 90 degrees."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor, seq_offset: int = 0) -> tuple[Tensor, Tensor]:
        """
        Apply RoPE to queries and keys.

        Args:
            q: (..., seq_len, head_dim)
            k: (..., seq_len, head_dim)
            seq_offset: for KV-cache / chunked generation
        Returns:
            q_rot, k_rot: same shapes as inputs
        """
        seq_len = q.shape[-2]
        if seq_offset + seq_len > self.max_seq_len:
            # Dynamically extend the cache
            self._build_cache(seq_offset + seq_len)

        cos = self.cos_cache[seq_offset : seq_offset + seq_len]  # (L, head_dim)
        sin = self.sin_cache[seq_offset : seq_offset + seq_len]

        # Broadcast over batch and head dims
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (causal, FlashAttention2-compatible)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.

    Uses torch.nn.functional.scaled_dot_product_attention which dispatches
    to FlashAttention2 automatically when:
      - CUDA device
      - Causal mask requested
      - Head dim <= 128

    Args:
        d_model:   model dimension
        num_heads: number of attention heads
        dropout:   attention dropout probability (0 for inference)
        use_rope:  apply Rotary Position Embedding (default: True)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope: Optional[RotaryPositionEmbedding] = None
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim, max_seq_len=max_seq_len
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[dict[str, Tensor]] = None,
        seq_offset: int = 0,
    ) -> tuple[Tensor, Optional[dict[str, Tensor]]]:
        """
        Args:
            x:          (B, L, d_model)
            kv_cache:   dict with keys "k", "v" — tensors of shape
                        (B, num_heads, L_cached, head_dim)
            seq_offset: position offset for RoPE + KV-cache
        Returns:
            out:      (B, L, d_model)
            kv_cache: updated dict (or None if not provided)
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, L, head_dim)
        def _reshape(t: Tensor) -> Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        # Apply RoPE
        if self.rope is not None:
            q, k = self.rope(q, k, seq_offset=seq_offset)

        # Append to KV cache (for autoregressive generation)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache = {"k": k.detach(), "v": v.detach()}

        # scaled_dot_product_attention → FlashAttention2 dispatch when available
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,      # is_causal handles the mask
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, num_heads, L, head_dim)

        # Merge heads
        out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)
        return out, kv_cache


# ---------------------------------------------------------------------------
# Multi-Head Cross-Attention (action conditioning)
# ---------------------------------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention where video tokens attend to action context vectors.

    Used to condition the transformer on forklift actions:
      - linear velocity (vx, vy)
      - angular velocity (omega)
      - fork height
      - fork tilt angle
      - load weight

    Args:
        d_model:        query dimension (transformer hidden size)
        context_dim:    key/value dimension (action embedding size)
        num_heads:      attention heads
        dropout:        attention dropout
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(context_dim, d_model, bias=False)
        self.v_proj = nn.Linear(context_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       (B, L_q, d_model)   — video token queries
            context: (B, L_kv, context_dim) — action conditioning
        Returns:
            out: (B, L_q, d_model)
        """
        B, L_q, _ = x.shape
        L_kv = context.shape[1]

        def _reshape_q(t: Tensor) -> Tensor:
            return t.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

        def _reshape_kv(t: Tensor) -> Tensor:
            return t.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        q = _reshape_q(self.q_proj(x))
        k = _reshape_kv(self.k_proj(context))
        v = _reshape_kv(self.v_proj(context))

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # cross-attention is non-causal
        )  # (B, num_heads, L_q, head_dim)

        out = attn_out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward network (SwiGLU variant)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network.
    FFN dim = 4 × d_model (standard), split as gate + value.
    """

    def __init__(self, d_model: int, ffn_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        ffn_dim = ffn_dim or int(d_model * 8 / 3)  # SwiGLU conventional ratio
        # Round up to next multiple of 64 for efficiency
        ffn_dim = ((ffn_dim + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)   # gate
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)   # value
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)   # down-projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: swish(gate) * value
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.dropout(self.w2(gate * value))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer block:
        LayerNorm → CausalSelfAttention → residual
        LayerNorm → CrossAttention (optional) → residual
        LayerNorm → FFN → residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        self.cross_attn: Optional[MultiHeadCrossAttention] = None
        if context_dim is not None:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadCrossAttention(
                d_model=d_model,
                context_dim=context_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        kv_cache: Optional[dict[str, Tensor]] = None,
        seq_offset: int = 0,
    ) -> tuple[Tensor, Optional[dict[str, Tensor]]]:
        """
        Args:
            x:        (B, L, d_model)
            context:  (B, L_ctx, context_dim) action embeddings (optional)
            kv_cache: dict for autoregressive generation
            seq_offset: KV cache offset
        Returns:
            x:        (B, L, d_model)
            kv_cache: updated cache
        """
        # Self-attention
        h, kv_cache = self.self_attn(self.norm1(x), kv_cache=kv_cache, seq_offset=seq_offset)
        x = x + h

        # Cross-attention (action conditioning)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)

        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x, kv_cache
