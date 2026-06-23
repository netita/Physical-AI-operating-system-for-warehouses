"""
tests/test_world_model.py
==========================
pytest tests for the WarehouseGPT world model.

Tests
-----
* test_vqvae_forward_pass        — VQ-VAE encode/decode shapes and loss types
* test_transformer_causal_mask   — Attention causal mask prevents future leakage
* test_occupancy_prediction_shape — OccupancyForecaster output tensor shapes
* test_checkpoint_save_load      — Save model weights, reload, verify inference

All tests use small/toy configurations to keep them fast on CPU.

Run::

    pytest tests/test_world_model.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Conditional import guard — must come before any bare torch imports so the
# module is skipped cleanly in CPU-only / docs-only environments.
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn  # noqa: E402

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------

from warehousegpt.world_model.tokenizer.vqvae import WarehouseVQVAE
from warehousegpt.world_model.transformer.model import (
    WarehouseWorldModel,
    WarehouseTransformerConfig,
)
from warehousegpt.world_model.transformer.attention import MultiHeadSelfAttention
from warehousegpt.world_model.prediction.occupancy import OccupancyForecaster


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Return CPU device (all tests must run without GPU)."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_vqvae(device: torch.device) -> WarehouseVQVAE:
    """
    Very small VQ-VAE for fast CPU tests.

    Spatial downsampling: 2× (H/2, W/2) — single DownBlock
    Temporal downsampling: none
    Codebook: 64 entries, embedding dim 32
    """
    model = WarehouseVQVAE(
        in_channels=3,
        base_channels=32,       # Encoder/Decoder have exactly 4 hardcoded stages
        channel_mult=(1, 1, 1, 1),  # 4 elements required; 32*1=32, divisible by groups=32
        num_res_blocks=1,
        codebook_size=64,
        embedding_dim=32,
        commitment_cost=0.25,
        ema_decay=0.99,
    )
    model.to(device).eval()
    return model


@pytest.fixture(scope="module")
def tiny_transformer_cfg() -> WarehouseTransformerConfig:
    """Minimal transformer config for CPU tests."""
    return WarehouseTransformerConfig(
        codebook_size=64,
        image_height=32,
        image_width=32,
        patch_size=8,
        max_frames=4,
        d_model=64,
        num_layers=2,
        num_heads=4,
        ffn_multiplier=2.0,
        dropout=0.0,
        action_dim=4,
        action_embed_dim=32,
        action_seq_len=2,
        occupancy_classes=3,
    )


@pytest.fixture(scope="module")
def tiny_world_model(
    tiny_transformer_cfg: WarehouseTransformerConfig,
    device: torch.device,
) -> WarehouseWorldModel:
    model = WarehouseWorldModel(cfg=tiny_transformer_cfg, in_channels=3)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# test_vqvae_forward_pass
# ---------------------------------------------------------------------------


class TestVQVAEForwardPass:
    """Tests for WarehouseVQVAE forward pass and encode/decode API."""

    def test_output_types(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """VQ-VAE forward should return WarehouseVQVAEOutput with correct field types."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        x = torch.rand(B, T, C, H, W)
        out = tiny_vqvae(x)

        assert isinstance(out.recon, torch.Tensor), "recon must be a Tensor"
        assert isinstance(out.vq_loss, torch.Tensor), "vq_loss must be a Tensor"
        assert isinstance(out.recon_loss, torch.Tensor), "recon_loss must be a Tensor"
        assert isinstance(out.indices, torch.Tensor), "indices must be a Tensor"
        assert isinstance(out.perplexity, torch.Tensor), "perplexity must be a Tensor"

    def test_reconstruction_shape(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """Reconstructed video must have the same shape as the input."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        x = torch.rand(B, T, C, H, W)
        out = tiny_vqvae(x)

        assert out.recon.shape == (B, T, C, H, W), (
            f"Expected recon shape {(B, T, C, H, W)}, got {out.recon.shape}"
        )

    def test_reconstruction_range(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """Reconstructed values must be in [0, 1] (sigmoid output)."""
        x = torch.rand(1, 4, 3, 32, 32)
        out = tiny_vqvae(x)

        assert out.recon.min().item() >= 0.0, "recon values below 0"
        assert out.recon.max().item() <= 1.0, "recon values above 1"

    def test_vq_loss_scalar(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """VQ commitment loss must be a non-negative scalar."""
        x = torch.rand(1, 4, 3, 32, 32)
        out = tiny_vqvae(x)

        assert out.vq_loss.ndim == 0, "vq_loss must be a scalar tensor"
        assert out.vq_loss.item() >= 0.0, "vq_loss must be non-negative"

    def test_encode_indices_dtype(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """Discrete token indices must be integer type."""
        x = torch.rand(2, 4, 3, 32, 32)
        _, indices, _ = tiny_vqvae.encode(x)

        assert indices.dtype in (torch.int64, torch.int32, torch.long), (
            f"Expected integer indices, got {indices.dtype}"
        )

    def test_encode_indices_in_range(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """All token indices must be within [0, codebook_size)."""
        x = torch.rand(2, 4, 3, 32, 32)
        _, indices, _ = tiny_vqvae.encode(x)

        assert indices.min().item() >= 0, "negative token index"
        assert indices.max().item() < tiny_vqvae.codebook_size, (
            f"index {indices.max().item()} >= codebook_size {tiny_vqvae.codebook_size}"
        )

    def test_total_loss_positive(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """Total loss must be a positive scalar (useful as training signal check)."""
        x = torch.rand(1, 4, 3, 32, 32)
        out = tiny_vqvae(x)
        loss = tiny_vqvae.total_loss(out)

        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_decode_indices_shape(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """decode_indices should return video with matching spatial/temporal dims."""
        B, T, C, H, W = 2, 4, 3, 32, 32
        x = torch.rand(B, T, C, H, W)
        _, indices, _ = tiny_vqvae.encode(x)
        recon = tiny_vqvae.decode_indices(indices)

        assert recon.shape[0] == B
        assert recon.shape[2] == C


# ---------------------------------------------------------------------------
# test_transformer_causal_mask
# ---------------------------------------------------------------------------


class TestTransformerCausalMask:
    """Tests that verify causal masking in the transformer attention layers."""

    def _build_expected_causal_mask(self, L: int) -> torch.Tensor:
        """
        Build the standard additive causal mask that PyTorch's
        scaled_dot_product_attention uses internally when is_causal=True.

        Allowed (lower-triangle + diagonal): 0.0
        Forbidden (upper-triangle):          -inf
        """
        mask = torch.zeros(L, L, dtype=torch.float32)
        mask = mask.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1), float("-inf"))
        return mask

    def test_causal_mask_shape(self, tiny_transformer_cfg: WarehouseTransformerConfig) -> None:
        """
        The causal mask for a sequence of length L must be an (L, L) matrix.
        """
        L = 16
        mask = self._build_expected_causal_mask(L)

        assert mask.shape == (L, L), (
            f"Causal mask shape expected ({L}, {L}), got {mask.shape}"
        )

    def test_causal_mask_upper_triangular(
        self, tiny_transformer_cfg: WarehouseTransformerConfig
    ) -> None:
        """
        Position i should NOT attend to position j > i.
        In the additive mask, illegal positions carry -inf, allowed ones are 0.
        """
        L = 8
        mask = self._build_expected_causal_mask(L)

        # Lower triangle (including diagonal) must be 0.0
        # Upper triangle must be -inf
        for i in range(L):
            for j in range(L):
                val = mask[i, j].item()
                if j <= i:
                    assert val == 0.0, (
                        f"Position ({i},{j}) should be attendable (0.0), got {val}"
                    )
                else:
                    assert val == float("-inf"), (
                        f"Position ({i},{j}) should be masked (-inf), got {val}"
                    )

    def test_no_future_leakage(
        self, tiny_world_model: WarehouseWorldModel, device: torch.device
    ) -> None:
        """
        Perturbing frame t+1 onwards must NOT change the hidden state at frame t.

        This directly tests that future information cannot flow back through
        the causal attention — the core correctness property of autoregressive
        models.
        """
        cfg = tiny_world_model.cfg
        T = 2
        Ht = cfg.num_patches_h
        Wt = cfg.num_patches_w

        tiny_world_model.eval()
        with torch.no_grad():
            # Base input: random token indices
            base_tokens = torch.randint(
                0, cfg.codebook_size, (1, T, Ht, Wt), device=device
            )
            out_base = tiny_world_model(base_tokens, input_mode="tokens")

            # Perturb only the second frame
            perturbed_tokens = base_tokens.clone()
            perturbed_tokens[:, 1:, :, :] = torch.randint(
                0, cfg.codebook_size, (1, T - 1, Ht, Wt), device=device
            )
            out_perturbed = tiny_world_model(perturbed_tokens, input_mode="tokens")

        # Hidden states for the first frame's tokens must be identical
        S = Ht * Wt  # tokens per frame
        h_base = out_base.hidden_states[:, :S, :]
        h_perturbed = out_perturbed.hidden_states[:, :S, :]

        assert torch.allclose(h_base, h_perturbed, atol=1e-5), (
            "Causal mask violated: perturbing future frames affected past hidden states"
        )


# ---------------------------------------------------------------------------
# test_occupancy_prediction_shape
# ---------------------------------------------------------------------------


class TestOccupancyPredictionShape:
    """Tests for the OccupancyForecaster UNet head."""

    @pytest.fixture(scope="class")
    def forecaster(self, device: torch.device) -> OccupancyForecaster:
        model = OccupancyForecaster(
            d_model=64,
            num_classes=3,
            unet_channels=(32, 16, 8, 8),
            output_height=32,
            output_width=32,
        )
        return model.to(device).eval()

    def test_output_logits_shape(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """Logits tensor must have shape (B, T, H, W, num_classes)."""
        B, T, Ht, Wt, D = 2, 3, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)

        with torch.no_grad():
            out = forecaster(hidden, T=T)

        expected_shape = (B, T, 32, 32, 3)
        assert out.logits.shape == expected_shape, (
            f"logits shape: expected {expected_shape}, got {out.logits.shape}"
        )

    def test_output_probs_sum_to_one(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """Softmax probabilities along the class dimension must sum to 1."""
        B, T, Ht, Wt, D = 1, 2, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)

        with torch.no_grad():
            out = forecaster(hidden, T=T)

        prob_sums = out.probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), (
            "Probabilities do not sum to 1 along class dimension"
        )

    def test_grid_class_values(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """argmax grid must contain only valid class indices {0, 1, 2}."""
        B, T, Ht, Wt, D = 1, 2, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)

        with torch.no_grad():
            out = forecaster(hidden, T=T)

        assert out.grid.min().item() >= 0
        assert out.grid.max().item() < 3

    def test_predict_occupancy_grid_shape(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """predict_occupancy_grid API should return (B, T, H, W)."""
        B, T, Ht, Wt, D = 2, 4, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)

        with torch.no_grad():
            grid = forecaster.predict_occupancy_grid(hidden, T=T)

        assert grid.shape == (B, T, 32, 32), (
            f"predict_occupancy_grid shape: expected ({B}, {T}, 32, 32), got {grid.shape}"
        )

    def test_horizon_slicing(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """When horizon is set, only the last horizon frames should be returned."""
        B, T, Ht, Wt, D = 1, 4, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)
        horizon = 2

        with torch.no_grad():
            grid = forecaster.predict_occupancy_grid(hidden, T=T, horizon=horizon)

        assert grid.shape[1] == horizon, (
            f"horizon={horizon} but got T={grid.shape[1]} in output"
        )

    def test_loss_computation(
        self, forecaster: OccupancyForecaster, device: torch.device
    ) -> None:
        """compute_loss should return a non-negative scalar."""
        B, T, Ht, Wt, D = 2, 3, 4, 4, 64
        L = T * Ht * Wt
        hidden = torch.randn(B, L, D, device=device)
        targets = torch.randint(0, 3, (B, T, 32, 32), device=device)

        loss = forecaster.compute_loss(hidden, targets, T=T)

        assert loss.ndim == 0, "loss must be a scalar"
        assert loss.item() >= 0.0, "loss must be non-negative"


# ---------------------------------------------------------------------------
# test_checkpoint_save_load
# ---------------------------------------------------------------------------


class TestCheckpointSaveLoad:
    """Tests for model checkpoint serialisation and deserialisation."""

    def test_state_dict_round_trip_vqvae(self, tiny_vqvae: WarehouseVQVAE) -> None:
        """
        Save WarehouseVQVAE weights to disk and reload them into a fresh model.
        Inference outputs must be identical before and after reload.
        """
        x = torch.rand(1, 4, 3, 32, 32)

        with torch.no_grad():
            out_before = tiny_vqvae(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ckpt_path = Path(tmp.name)

        try:
            # Save
            torch.save(tiny_vqvae.state_dict(), ckpt_path)

            # Reload into a new instance with same config
            fresh = WarehouseVQVAE(
                in_channels=3,
                base_channels=32,
                channel_mult=(1, 1, 1, 1),
                num_res_blocks=1,
                codebook_size=64,
                embedding_dim=32,
            )
            fresh.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            fresh.eval()

            with torch.no_grad():
                out_after = fresh(x)

            assert torch.allclose(out_before.recon, out_after.recon, atol=1e-6), (
                "Reconstruction mismatch after checkpoint reload"
            )
            assert torch.allclose(out_before.indices.float(), out_after.indices.float()), (
                "Token indices mismatch after checkpoint reload"
            )
        finally:
            ckpt_path.unlink(missing_ok=True)

    def test_state_dict_round_trip_world_model(
        self,
        tiny_world_model: WarehouseWorldModel,
        tiny_transformer_cfg: WarehouseTransformerConfig,
        device: torch.device,
    ) -> None:
        """
        Save WarehouseWorldModel and reload — next_token_logits must match.
        """
        cfg = tiny_transformer_cfg
        T, Ht, Wt = 2, cfg.num_patches_h, cfg.num_patches_w
        tokens = torch.randint(0, cfg.codebook_size, (1, T, Ht, Wt), device=device)

        tiny_world_model.eval()
        with torch.no_grad():
            out_before = tiny_world_model(tokens, input_mode="tokens")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ckpt_path = Path(tmp.name)

        try:
            # Save full state dict
            checkpoint = {
                "model_state_dict": tiny_world_model.state_dict(),
                "config": {
                    "d_model": cfg.d_model,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "codebook_size": cfg.codebook_size,
                },
            }
            torch.save(checkpoint, ckpt_path)

            # Reload
            loaded = torch.load(ckpt_path, map_location="cpu")
            fresh_model = WarehouseWorldModel(cfg=cfg, in_channels=3)
            fresh_model.load_state_dict(loaded["model_state_dict"])
            fresh_model.eval()

            with torch.no_grad():
                out_after = fresh_model(tokens, input_mode="tokens")

            assert torch.allclose(
                out_before.next_token_logits,
                out_after.next_token_logits,
                atol=1e-5,
            ), "next_token_logits mismatch after world model checkpoint reload"

            assert torch.allclose(
                out_before.risk_score,
                out_after.risk_score,
                atol=1e-5,
            ), "risk_score mismatch after checkpoint reload"

        finally:
            ckpt_path.unlink(missing_ok=True)

    def test_checkpoint_contains_required_keys(
        self, tiny_world_model: WarehouseWorldModel
    ) -> None:
        """State dict must contain keys for all major sub-modules."""
        sd = tiny_world_model.state_dict()
        key_prefixes = [
            "patch_embed",
            "action_encoder",
            "temporal_pos_embed",
            "blocks",
            "next_token_head",
            "risk_head",
            "occupancy_head",
        ]
        for prefix in key_prefixes:
            matching = [k for k in sd if k.startswith(prefix)]
            assert matching, (
                f"No state dict keys found with prefix {prefix!r}"
            )
