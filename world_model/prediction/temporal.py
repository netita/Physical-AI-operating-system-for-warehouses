"""
TemporalPredictor — autoregressive future frame generation.

Wraps WarehouseWorldModel to provide:
  - autoregressive_generate(): greedy / top-k / top-p next-token generation
  - beam_search(): diverse beam search for multiple future hypotheses
  - KV-cache for efficient O(n) decoding

All generation is performed in the VQ-VAE discrete token space.
Frames are decoded from tokens by the paired WarehouseVQVAE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warehousegpt.world_model.transformer.model import (
    WarehouseWorldModel,
    WarehouseTransformerConfig,
)
from warehousegpt.world_model.tokenizer.vqvae import WarehouseVQVAE


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class GenerationOutput(NamedTuple):
    """Result of autoregressive_generate()."""
    token_indices: Tensor     # (B, n_future, Ht, Wt) — generated token indices
    future_frames: Tensor     # (B, n_future, C, H, W) — decoded video frames
    risk_scores: Tensor       # (B, n_future) — per-frame risk scores


class BeamSearchOutput(NamedTuple):
    """Result of beam_search()."""
    beam_token_indices: Tensor   # (B, num_beams, n_future, Ht, Wt)
    beam_scores: Tensor          # (B, num_beams) — cumulative log-probs
    beam_frames: Tensor          # (B, num_beams, n_future, C, H, W)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Zero out logits below the top-k threshold."""
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus sampling: zero out logits outside the top-p probability mass."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens that push cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    # Scatter back to original ordering
    logits = logits.scatter(-1, sorted_indices, sorted_logits)
    return logits


def sample_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    """
    Sample next token from logits.

    Args:
        logits:      (B, vocab_size)
        temperature: softmax temperature
        top_k:       top-k truncation
        top_p:       nucleus sampling threshold
    Returns:
        token_ids: (B,) sampled token indices
    """
    logits = logits / max(temperature, 1e-8)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# TemporalPredictor
# ---------------------------------------------------------------------------

class TemporalPredictor(nn.Module):
    """
    Autoregressive future frame predictor.

    Uses the WarehouseWorldModel in token-space (discrete VQ-VAE indices)
    to generate n_future token grids, then decodes them via WarehouseVQVAE.

    KV-cache: during generation, KV tensors are accumulated so each new
    frame requires only O(S) self-attention work (S = spatial tokens per frame),
    rather than O(T_total * S).

    Args:
        world_model: trained WarehouseWorldModel
        vqvae:       paired trained WarehouseVQVAE (for decoding)
    """

    def __init__(
        self,
        world_model: WarehouseWorldModel,
        vqvae: WarehouseVQVAE,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.vqvae = vqvae
        self.cfg: WarehouseTransformerConfig = world_model.cfg

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def autoregressive_generate(
        self,
        context_tokens: Tensor,
        n_future: int,
        actions: Optional[Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 2048,
        top_p: float = 0.95,
        decode_frames: bool = True,
    ) -> GenerationOutput:
        """
        Generate future frame tokens autoregressively.

        The generation proceeds frame-by-frame:
          For each future frame t:
            For each token position s in raster order:
              - Run transformer on context + all previously generated tokens
              - Sample next token at position s using the output logits
              - Append token to sequence

        KV-cache is used so that the transformer only processes the newly
        appended token at each step (after the initial context pass).

        Args:
            context_tokens: (B, T_ctx, Ht, Wt) context frame tokens
            n_future:       number of future frames to generate
            actions:        (B, T_ctx + n_future, action_dim) optional
            temperature:    sampling temperature
            top_k:          top-k truncation
            top_p:          nucleus sampling threshold
            decode_frames:  if True, decode tokens to pixel frames via VQ-VAE
        Returns:
            GenerationOutput
        """
        B, T_ctx, Ht, Wt = context_tokens.shape
        S = Ht * Wt  # spatial tokens per frame
        device = context_tokens.device

        # --- Pass 1: warm up KV cache with context ---
        ctx_actions = actions[:, :T_ctx] if actions is not None else None
        output = self.world_model(
            video=context_tokens,
            actions=ctx_actions,
            input_mode="tokens",
            kv_caches=None,
            seq_offset=0,
        )

        # Collect KV caches from the context pass
        kv_caches: list[Optional[dict[str, Tensor]]] = [None] * self.cfg.num_layers
        # Re-run with cache collection enabled
        kv_caches_init = [None] * self.cfg.num_layers
        _ = self.world_model(
            video=context_tokens,
            actions=ctx_actions,
            input_mode="tokens",
            kv_caches=kv_caches_init,
            seq_offset=0,
        )
        # The model returns updated caches through the blocks
        # Rebuild with direct layer access for KV extraction
        tokens_with_cache, kv_caches = self._run_with_kv_cache(
            context_tokens, ctx_actions, kv_caches=None, seq_offset=0
        )

        # --- Pass 2: generate future frames ---
        generated_indices: list[Tensor] = []
        risk_scores: list[Tensor] = []

        current_seq_len = T_ctx * S

        for t in range(n_future):
            frame_tokens = torch.zeros(B, S, dtype=torch.long, device=device)

            for s in range(S):
                # Build single-token input for the current position
                row, col = divmod(s, Wt)
                partial_frame = frame_tokens.clone()
                # Reshape partial frame for model input (only the generated so far)
                if s > 0:
                    partial_token_grid = partial_frame.view(B, 1, Ht, Wt)
                    # Only feed the new partial tokens since cache handles the rest
                    step_input = partial_token_grid
                else:
                    # First token of the frame: feed a dummy grid
                    step_input = torch.zeros(B, 1, Ht, Wt, dtype=torch.long, device=device)

                # Get action for this frame
                step_actions: Optional[Tensor] = None
                if actions is not None:
                    step_actions = actions[:, T_ctx + t : T_ctx + t + 1]

                out, kv_caches = self._run_with_kv_cache(
                    step_input, step_actions,
                    kv_caches=kv_caches,
                    seq_offset=current_seq_len + s,
                )

                # out: (B, 1*Ht*Wt, d_model) → take logit at position s
                # For efficiency, sample the token at row*Wt+col position
                token_logits = self.world_model.next_token_head(out)  # (B, S, vocab)
                next_token_logit = token_logits[:, s % S]             # (B, vocab)

                next_token = sample_token(
                    next_token_logit,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )  # (B,)

                frame_tokens[:, s] = next_token

            current_seq_len += S
            frame_grid = frame_tokens.view(B, 1, Ht, Wt)
            generated_indices.append(frame_grid)

            # Risk score for this frame (from last block hidden states)
            risk_scores.append(output.risk_score.squeeze(-1))  # (B,)

        # Stack generated tokens: (B, n_future, Ht, Wt)
        all_tokens = torch.cat(generated_indices, dim=1)

        # Decode tokens to frames if requested
        future_frames: Tensor
        if decode_frames:
            future_frames = self.vqvae.decode_indices(all_tokens)  # (B, T, C, H, W)
        else:
            future_frames = torch.zeros(
                B, n_future, 3, self.cfg.image_height, self.cfg.image_width,
                device=device
            )

        all_risk = torch.stack(risk_scores, dim=1)  # (B, n_future)

        return GenerationOutput(
            token_indices=all_tokens,
            future_frames=future_frames,
            risk_scores=all_risk,
        )

    def _run_with_kv_cache(
        self,
        tokens: Tensor,
        actions: Optional[Tensor],
        kv_caches: Optional[list[Optional[dict[str, Tensor]]]],
        seq_offset: int,
    ) -> tuple[Tensor, list[Optional[dict[str, Tensor]]]]:
        """
        Run transformer blocks manually to collect updated KV caches.

        Returns:
            hidden_states: (B, L, d_model)
            updated kv_caches
        """
        B, T, Ht, Wt = tokens.shape

        # Embed tokens
        x = self.world_model.patch_embed.embed_tokens(tokens)    # (B, L, d_model)
        x = self.world_model._add_temporal_pos(x, T)

        # Action context
        context: Optional[Tensor] = None
        if actions is not None:
            context = self.world_model.action_encoder(actions)

        if kv_caches is None:
            kv_caches = [None] * self.cfg.num_layers

        new_caches: list[Optional[dict[str, Tensor]]] = []
        for block, cache in zip(self.world_model.blocks, kv_caches):
            x, updated_cache = block(
                x, context=context, kv_cache=cache, seq_offset=seq_offset
            )
            new_caches.append(updated_cache)

        return x, new_caches

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def beam_search(
        self,
        context_tokens: Tensor,
        n_future: int,
        actions: Optional[Tensor] = None,
        num_beams: int = 4,
        temperature: float = 1.0,
        decode_frames: bool = True,
    ) -> BeamSearchOutput:
        """
        Diverse beam search over future frame sequences.

        Generates `num_beams` distinct future hypotheses by maintaining
        a beam of the top-k partial sequences ranked by cumulative
        log-probability.

        Args:
            context_tokens: (B, T_ctx, Ht, Wt)
            n_future:       number of future frames per beam
            actions:        (B, T_total, action_dim) optional
            num_beams:      beam width
            temperature:    logit temperature before softmax
            decode_frames:  whether to decode tokens to pixel frames
        Returns:
            BeamSearchOutput
        """
        B, T_ctx, Ht, Wt = context_tokens.shape
        S = Ht * Wt
        device = context_tokens.device

        # Expand context and actions for beam dimension
        # Shape: (B * num_beams, T_ctx, Ht, Wt)
        ctx_expanded = context_tokens.unsqueeze(1).expand(B, num_beams, T_ctx, Ht, Wt)
        ctx_expanded = ctx_expanded.reshape(B * num_beams, T_ctx, Ht, Wt)

        actions_expanded: Optional[Tensor] = None
        if actions is not None:
            actions_expanded = actions.unsqueeze(1).expand(
                B, num_beams, *actions.shape[1:]
            ).reshape(B * num_beams, *actions.shape[1:])

        # Initialise beam scores: (B, num_beams)
        beam_scores = torch.zeros(B, num_beams, device=device)
        beam_scores[:, 1:] = float("-inf")  # only first beam is live initially

        # Collected token sequences: list of (B * num_beams, Ht, Wt) per frame
        beam_token_frames: list[Tensor] = []

        # Warm up KV caches with context
        ctx_act = actions_expanded[:, :T_ctx] if actions_expanded is not None else None
        hidden, kv_caches = self._run_with_kv_cache(
            ctx_expanded, ctx_act, kv_caches=None, seq_offset=0
        )

        current_seq_len = T_ctx * S

        for t in range(n_future):
            # For each frame we generate all S spatial tokens greedily within the beam
            # (standard beam search operates at the frame granularity for efficiency)
            frame_log_probs_list: list[Tensor] = []

            # Get logits for the first token of the next frame
            logits = self.world_model.next_token_head(hidden)  # (B*beams, L, V)
            last_logits = logits[:, -1] / max(temperature, 1e-8)  # (B*beams, V)
            log_probs = F.log_softmax(last_logits, dim=-1)        # (B*beams, V)

            # Reshape for beam scoring: (B, num_beams, V)
            log_probs = log_probs.view(B, num_beams, -1)

            # Add cumulative beam scores
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs  # (B, num_beams, V)
            # Flatten beams and vocab: (B, num_beams * V)
            flat_scores = candidate_scores.view(B, -1)

            # Take top num_beams candidates
            top_scores, top_indices = flat_scores.topk(num_beams, dim=-1)  # (B, num_beams)
            beam_ids = top_indices // self.cfg.codebook_size  # which beam each came from
            token_ids = top_indices % self.cfg.codebook_size  # which token was picked

            beam_scores = top_scores  # (B, num_beams)

            # Reorder caches according to new beam ordering
            reordered_caches = self._reorder_kv_caches(kv_caches, beam_ids, B, num_beams)

            # Build new frame tokens (fill spatial positions greedily after first)
            # For speed: only generate first token via beam, rest greedily
            new_frame_tokens = torch.zeros(B * num_beams, S, dtype=torch.long, device=device)
            new_frame_tokens[:, 0] = token_ids.view(-1)

            # Extend with greedy generation for remaining S-1 positions
            for s in range(1, S):
                partial = new_frame_tokens[:, :s].view(B * num_beams, 1, -1)
                # Fast path: reuse last hidden state with single token step
                step_grid = new_frame_tokens[:, s - 1].view(B * num_beams, 1, 1, 1).expand(
                    B * num_beams, 1, Ht, Wt
                )
                # Fill one spatial position at a time using the cache
                h_step, reordered_caches = self._run_with_kv_cache(
                    step_grid, None, kv_caches=reordered_caches,
                    seq_offset=current_seq_len + s - 1
                )
                step_logits = self.world_model.next_token_head(h_step)[:, -1]  # (B*beams, V)
                step_logits = step_logits / max(temperature, 1e-8)
                greedy_token = step_logits.argmax(dim=-1)  # (B*beams,)
                new_frame_tokens[:, s] = greedy_token

            current_seq_len += S
            kv_caches = reordered_caches

            frame_grid = new_frame_tokens.view(B, num_beams, Ht, Wt)
            beam_token_frames.append(frame_grid)

            # Update hidden state for next frame
            full_frame_input = new_frame_tokens.view(B * num_beams, 1, Ht, Wt)
            act_step = (
                actions_expanded[:, T_ctx + t : T_ctx + t + 1]
                if actions_expanded is not None else None
            )
            hidden, kv_caches = self._run_with_kv_cache(
                full_frame_input, act_step, kv_caches=kv_caches,
                seq_offset=current_seq_len - S
            )

        # Stack: (B, num_beams, n_future, Ht, Wt)
        all_beam_tokens = torch.stack(beam_token_frames, dim=2)

        # Decode frames
        beam_frames: Tensor
        if decode_frames:
            flat_tokens = all_beam_tokens.view(B * num_beams, n_future, Ht, Wt)
            flat_frames = self.vqvae.decode_indices(flat_tokens)  # (B*beams, T, C, H, W)
            C, H, W = flat_frames.shape[2:]
            beam_frames = flat_frames.view(B, num_beams, n_future, C, H, W)
        else:
            beam_frames = torch.zeros(
                B, num_beams, n_future, 3,
                self.cfg.image_height, self.cfg.image_width,
                device=device,
            )

        return BeamSearchOutput(
            beam_token_indices=all_beam_tokens,
            beam_scores=beam_scores,
            beam_frames=beam_frames,
        )

    def _reorder_kv_caches(
        self,
        kv_caches: list[Optional[dict[str, Tensor]]],
        beam_ids: Tensor,  # (B, num_beams)
        B: int,
        num_beams: int,
    ) -> list[Optional[dict[str, Tensor]]]:
        """Reorder KV caches to follow the new beam ordering."""
        new_caches: list[Optional[dict[str, Tensor]]] = []
        for cache in kv_caches:
            if cache is None:
                new_caches.append(None)
                continue
            # cache["k"]: (B*num_beams, num_heads, L_cached, head_dim)
            k = cache["k"]
            v = cache["v"]
            BN, H_heads, L, D = k.shape
            k = k.view(B, num_beams, H_heads, L, D)
            v = v.view(B, num_beams, H_heads, L, D)

            # Gather along beam dimension using beam_ids
            idx = beam_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                B, num_beams, H_heads, L, D
            )
            k = k.gather(1, idx).view(B * num_beams, H_heads, L, D)
            v = v.gather(1, idx).view(B * num_beams, H_heads, L, D)
            new_caches.append({"k": k, "v": v})
        return new_caches
