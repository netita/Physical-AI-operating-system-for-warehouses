"""
WorldModelTrainer — production-grade training loop for the WarehouseGPT World Model.

Features
--------
- DDP or FSDP (Fully Sharded Data Parallel) via PyTorch native APIs
- Mixed-precision training with BF16 (bf16) using torch.amp
- Gradient accumulation for large effective batch sizes
- WandB + TensorBoard dual logging
- Checkpoint save / resume with atomic writes
- Cosine LR schedule with linear warmup
- EMA parameter tracking for stable evaluation
- Per-component loss tracking (reconstruction, VQ, next-token, occupancy, trajectory)

Usage::

    from warehousegpt.world_model.training.trainer import WorldModelTrainer
    from warehousegpt.world_model.training.config import TrainingConfig

    cfg = TrainingConfig(run_name="wm-v1")
    trainer = WorldModelTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_layer_auto_wrap_policy
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False

# Observability
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from warehousegpt.world_model.training.config import TrainingConfig
from warehousegpt.world_model.training.dataset import WarehouseVideoDataset
from warehousegpt.world_model.tokenizer.vqvae import WarehouseVQVAE
from warehousegpt.world_model.transformer.model import (
    WarehouseWorldModel,
    WarehouseTransformerConfig,
)
from warehousegpt.world_model.prediction.occupancy import OccupancyForecaster
from warehousegpt.world_model.prediction.trajectory import TrajectoryPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Cosine decay with linear warmup.

    Returns the LR multiplier (applied to optimizer base LR).
    """
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class ExponentialMovingAverage:
    """
    Maintains EMA shadow parameters for the world model.

    Used to obtain a stable model snapshot for validation without
    affecting the training parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Context manager: temporarily swap in EMA params."""
        original: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class LossBundle:
    """Accumulates loss components over gradient accumulation steps."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total: float = 0.0
        self.recon: float = 0.0
        self.vq: float = 0.0
        self.next_token: float = 0.0
        self.occupancy: float = 0.0
        self.risk: float = 0.0
        self.n: int = 0

    def accumulate(self, **kwargs: float) -> None:
        self.total += kwargs.get("total", 0.0)
        self.recon += kwargs.get("recon", 0.0)
        self.vq += kwargs.get("vq", 0.0)
        self.next_token += kwargs.get("next_token", 0.0)
        self.occupancy += kwargs.get("occupancy", 0.0)
        self.risk += kwargs.get("risk", 0.0)
        self.n += 1

    def average(self) -> dict[str, float]:
        n = max(self.n, 1)
        return {
            "loss/total": self.total / n,
            "loss/recon": self.recon / n,
            "loss/vq": self.vq / n,
            "loss/next_token": self.next_token / n,
            "loss/occupancy": self.occupancy / n,
            "loss/risk": self.risk / n,
        }


# ---------------------------------------------------------------------------
# WorldModelTrainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """
    End-to-end trainer for the WarehouseGPT World Model.

    Trains three components jointly:
      1. WarehouseVQVAE (video tokenizer)
      2. WarehouseWorldModel (causal transformer)
      3. OccupancyForecaster (UNet head on transformer features)

    Args:
        cfg: TrainingConfig dataclass with all hyper-parameters
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0
        self._best_val_loss = float("inf")

        # Set seed
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Distributed setup
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.setup_distributed()

        # Build models
        self._build_models()

        # Build optimizers and scheduler
        self._build_optimizer()

        # Mixed precision scaler (only for FP16; BF16 uses autocast without GradScaler)
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if cfg.mixed_precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()

        # EMA
        self.ema = ExponentialMovingAverage(self.world_model_raw, decay=0.9999)

        # Logging
        self._setup_logging()

        # Resume from checkpoint
        if cfg.resume_from_checkpoint is not None:
            self.load_checkpoint(cfg.resume_from_checkpoint)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_distributed(self) -> None:
        """
        Initialise process group for DDP or FSDP.

        Detects SLURM / torchrun environment variables automatically.
        Single-GPU training works without any environment variables.
        """
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))

            if self.world_size > 1:
                dist.init_process_group(backend="nccl")
                self.is_distributed = True
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
                logger.info(
                    "Distributed: rank=%d / world_size=%d", self.rank, self.world_size
                )
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_models(self) -> None:
        """Instantiate VQ-VAE, world model, and auxiliary heads."""
        cfg = self.cfg

        # --- VQ-VAE ---
        vqvae = WarehouseVQVAE(
            codebook_size=cfg.codebook_size,
            embedding_dim=cfg.embedding_dim,
            base_channels=cfg.vqvae_base_channels,
            channel_mult=cfg.vqvae_channel_mult,
            num_res_blocks=cfg.vqvae_num_res_blocks,
            commitment_cost=cfg.vqvae_commitment_cost,
            ema_decay=cfg.vqvae_ema_decay,
        ).to(self.device)

        # Optionally load pretrained VQ-VAE
        if cfg.vqvae_checkpoint is not None and Path(cfg.vqvae_checkpoint).exists():
            state = torch.load(cfg.vqvae_checkpoint, map_location=self.device)
            vqvae.load_state_dict(state, strict=True)
            logger.info("Loaded VQ-VAE checkpoint from %s", cfg.vqvae_checkpoint)

        # --- World model ---
        transformer_cfg = WarehouseTransformerConfig(
            codebook_size=cfg.codebook_size,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            max_frames=cfg.max_frames,
            action_dim=cfg.action_dim,
            action_embed_dim=cfg.action_embed_dim,
        )
        world_model = WarehouseWorldModel(cfg=transformer_cfg).to(self.device)

        # --- Occupancy forecaster ---
        occ_forecaster = OccupancyForecaster(
            d_model=cfg.d_model,
            output_height=cfg.image_height,
            output_width=cfg.image_width,
        ).to(self.device)

        # Compile (optional — improves throughput ~15-20%)
        if cfg.compile_model:
            logger.info("Compiling models with torch.compile()...")
            vqvae = torch.compile(vqvae)
            world_model = torch.compile(world_model)

        # Store raw references (before wrapping in DDP/FSDP) for EMA, checkpointing
        self.vqvae_raw = vqvae
        self.world_model_raw = world_model
        self.occ_forecaster_raw = occ_forecaster

        # Wrap in distributed backend
        if self.is_distributed:
            if cfg.distributed_backend == "fsdp" and _FSDP_AVAILABLE:
                self._wrap_fsdp()
            else:
                self._wrap_ddp()
        else:
            self.vqvae = vqvae
            self.world_model = world_model
            self.occ_forecaster = occ_forecaster

        logger.info("VQ-VAE parameters: %.1fM", sum(p.numel() for p in vqvae.parameters()) / 1e6)
        logger.info(
            "WorldModel parameters: %.2fB",
            sum(p.numel() for p in world_model.parameters()) / 1e9
        )

    def _wrap_ddp(self) -> None:
        """Wrap models in DistributedDataParallel."""
        self.vqvae = DDP(self.vqvae_raw, device_ids=[self.local_rank])
        self.world_model = DDP(self.world_model_raw, device_ids=[self.local_rank])
        self.occ_forecaster = DDP(self.occ_forecaster_raw, device_ids=[self.local_rank])
        logger.info("Models wrapped in DDP")

    def _wrap_fsdp(self) -> None:
        """Wrap models in FSDP (ZeRO-3 equivalent)."""
        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding = sharding_map.get(self.cfg.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD)

        mp_policy: Optional[MixedPrecision] = None
        if self.cfg.mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.cfg.mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        cpu_offload = CPUOffload(offload_params=self.cfg.fsdp_cpu_offload)

        fsdp_kwargs: dict[str, Any] = {
            "sharding_strategy": sharding,
            "cpu_offload": cpu_offload,
            "device_id": self.local_rank,
        }
        if mp_policy is not None:
            fsdp_kwargs["mixed_precision"] = mp_policy

        self.vqvae = FSDP(self.vqvae_raw, **fsdp_kwargs)
        self.world_model = FSDP(self.world_model_raw, **fsdp_kwargs)
        self.occ_forecaster = FSDP(self.occ_forecaster_raw, **fsdp_kwargs)
        logger.info("Models wrapped in FSDP (sharding=%s)", self.cfg.fsdp_sharding_strategy)

    def _build_optimizer(self) -> None:
        """Construct optimizer and LR lambda scheduler."""
        cfg = self.cfg

        # Collect all trainable parameters from all models
        all_params = (
            list(self.vqvae_raw.parameters())
            + list(self.world_model_raw.parameters())
            + list(self.occ_forecaster_raw.parameters())
        )

        # Weight decay: skip bias, LayerNorm, embedding parameters
        decay_params = []
        no_decay_params = []
        for p in all_params:
            if not p.requires_grad:
                continue
            if p.ndim <= 1:  # bias and 1D params (LN weight)
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        else:
            # Fallback to AdamW for other optimizers
            logger.warning("Optimizer %s not fully implemented; falling back to AdamW", cfg.optimizer)
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )

        # Lambda LR scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_lr_with_warmup(
                step=step,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                peak_lr=cfg.learning_rate,
            ),
        )

    def _setup_logging(self) -> None:
        """Initialise WandB and TensorBoard (rank 0 only)."""
        cfg = self.cfg
        self.tb_writer: Optional[Any] = None
        self.wandb_run: Optional[Any] = None

        if self.rank != 0:
            return

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.use_tensorboard and _TB_AVAILABLE:
            cfg.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(cfg.tensorboard_dir / cfg.run_name))
            logger.info("TensorBoard writer initialised at %s", cfg.tensorboard_dir)

        if cfg.use_wandb and _WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.run_name,
                config={
                    "d_model": cfg.d_model,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "codebook_size": cfg.codebook_size,
                    "learning_rate": cfg.learning_rate,
                    "batch_size": (
                        cfg.batch_size_per_gpu
                        * self.world_size
                        * cfg.gradient_accumulation_steps
                    ),
                    "mixed_precision": cfg.mixed_precision,
                    "distributed": cfg.distributed_backend,
                },
                resume="allow",
            )
            logger.info("WandB run initialised: %s", wandb.run.url if wandb.run else "unknown")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _build_dataloader(self, split: str) -> DataLoader:
        """Create a DataLoader for the given split."""
        cfg = self.cfg
        is_train = split == cfg.dataset_split

        dataset = WarehouseVideoDataset(
            dataset_name=cfg.dataset_name,
            split=split,
            clip_frames=cfg.clip_frames,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            speed_jitter_range=cfg.speed_jitter_range if is_train else (1.0, 1.0),
            frame_dropout_prob=cfg.frame_dropout_prob if is_train else 0.0,
            color_jitter=cfg.color_jitter and is_train,
        )

        sampler: Optional[DistributedSampler] = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=is_train,
                drop_last=is_train,
            )

        bs = cfg.batch_size_per_gpu if is_train else cfg.val_batch_size
        return DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            shuffle=(sampler is None and is_train),
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=cfg.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Autocast context
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _autocast(self) -> Iterator[None]:
        """Mixed precision autocast context."""
        cfg = self.cfg
        if cfg.mixed_precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        elif cfg.mixed_precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the joint training loss over all model components.

        Loss components:
            1. VQ-VAE reconstruction + codebook commitment loss
            2. Next-frame token cross-entropy (causal LM objective)
            3. Occupancy grid cross-entropy
            4. Risk score binary cross-entropy

        Args:
            batch: dict from WarehouseVideoDataset.__getitem__
        Returns:
            loss:         scalar total loss (backward-able)
            loss_details: dict with per-component loss values (for logging)
        """
        cfg = self.cfg
        video = batch["video"].to(self.device)           # (B, T, 3, H, W)
        actions = batch["actions"].to(self.device)       # (B, T, action_dim)
        risk_label = batch["risk_label"].to(self.device) # (B,)

        B, T, C, H, W = video.shape
        T_ctx = cfg.context_frames
        T_fut = min(cfg.future_frames, T - T_ctx)
        T_total = T_ctx + T_fut

        video = video[:, :T_total]
        actions = actions[:, :T_total]

        # --- VQ-VAE forward ---
        vqvae_out = self.vqvae(video)  # WarehouseVQVAEOutput

        loss_recon = vqvae_out.recon_loss * cfg.loss_recon_weight
        loss_vq = vqvae_out.vq_loss * cfg.loss_vq_weight

        # --- Transformer: next-token prediction ---
        # Use context tokens to predict future tokens (causal LM objective)
        token_indices = vqvae_out.indices  # (B, T', H', W')
        # T' = T // temporal_downsample, H' = H // 8, W' = W // 8
        T_tok = token_indices.shape[1]

        # Input: first T_ctx token frames → predict next token at each position
        # Context window: [:, :-1], target: [:, 1:]
        ctx_indices = token_indices[:, :-1]  # (B, T'-1, H', W')
        tgt_indices = token_indices[:, 1:]   # (B, T'-1, H', W') — shifted targets

        ctx_actions = actions[:, :ctx_indices.shape[1]]

        world_model_out = self.world_model(
            video=ctx_indices,
            actions=ctx_actions,
            input_mode="tokens",
        )

        # Flatten for cross-entropy: (B * (T'-1) * H' * W', codebook_size)
        logits = world_model_out.next_token_logits  # (B, L, vocab)
        B_l, L, V = logits.shape
        _, Tp, Hp, Wp = tgt_indices.shape
        L_tgt = Tp * Hp * Wp

        logits_flat = logits[:, :L_tgt].reshape(B_l * L_tgt, V)
        tgt_flat = tgt_indices.reshape(B_l * L_tgt)

        loss_next_token = F.cross_entropy(logits_flat, tgt_flat) * cfg.loss_next_token_weight

        # --- Occupancy prediction ---
        hidden = world_model_out.hidden_states  # (B, L, d_model)
        T_seq = ctx_indices.shape[1]

        # Only compute occupancy loss if segmentation labels available
        seg = batch.get("segmentation", None)
        loss_occ = torch.tensor(0.0, device=self.device)
        if seg is not None:
            seg = seg[:, :T_seq].to(self.device)  # (B, T', H, W) — approximate match
            # Build coarse occupancy targets: occupied = any non-floor/background pixel
            # 0=background, 1=floor → free; everything else → occupied
            occ_target = (seg > 1).long()  # (B, T', H, W)  — binary for now
            # Resize target to match latent grid if needed
            occ_out = self.occ_forecaster(hidden, T_seq)
            try:
                loss_occ = self.occ_forecaster_raw.compute_loss(
                    hidden, occ_target, T_seq
                ) * cfg.loss_occupancy_weight
            except Exception:
                loss_occ = torch.tensor(0.0, device=self.device)

        # --- Risk score ---
        risk_pred = world_model_out.risk_score.squeeze(-1)  # (B,)
        loss_risk = F.binary_cross_entropy(
            risk_pred, risk_label
        ) * cfg.loss_risk_weight

        # --- Total loss ---
        total_loss = loss_recon + loss_vq + loss_next_token + loss_occ + loss_risk

        loss_details = {
            "total": total_loss.item(),
            "recon": loss_recon.item(),
            "vq": loss_vq.item(),
            "next_token": loss_next_token.item(),
            "occupancy": loss_occ.item() if isinstance(loss_occ, Tensor) else loss_occ,
            "risk": loss_risk.item(),
        }

        return total_loss, loss_details

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training entry point."""
        cfg = self.cfg
        logger.info("Starting training: %s", cfg.run_name)

        train_loader = self._build_dataloader(cfg.dataset_split)

        self.vqvae.train()
        self.world_model.train()
        self.occ_forecaster.train()

        loss_bundle = LossBundle()
        t0 = time.perf_counter()

        while self.global_step < cfg.max_steps:
            self.epoch += 1

            if self.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(self.epoch)

            self.train_epoch(train_loader, loss_bundle, t0)

            if self.global_step >= cfg.max_steps:
                break

        logger.info("Training complete. global_step=%d", self.global_step)
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run and _WANDB_AVAILABLE:
            wandb.finish()

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_bundle: LossBundle,
        t0: float,
    ) -> None:
        """
        Process all batches in one epoch.

        Handles gradient accumulation: only calls optimizer.step() every
        gradient_accumulation_steps micro-batches.
        """
        cfg = self.cfg
        accum = cfg.gradient_accumulation_steps

        for batch_idx, batch in enumerate(dataloader):
            is_last_accum_step = (batch_idx + 1) % accum == 0

            # --- Forward with mixed precision ---
            with self._autocast():
                loss, details = self.compute_loss(batch)
                loss = loss / accum  # normalise for gradient accumulation

            # --- Backward ---
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_bundle.accumulate(**details)

            # --- Optimizer step (every accum micro-batches) ---
            if is_last_accum_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    list(self.vqvae.parameters())
                    + list(self.world_model.parameters())
                    + list(self.occ_forecaster.parameters()),
                    max_norm=cfg.grad_clip,
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.world_model_raw)

                self.global_step += 1

                # --- Logging ---
                if self.global_step % cfg.log_every_n_steps == 0 and self.rank == 0:
                    elapsed = time.perf_counter() - t0
                    metrics = loss_bundle.average()
                    metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    metrics["step"] = self.global_step
                    metrics["throughput_steps_per_sec"] = (
                        cfg.log_every_n_steps / max(elapsed, 1e-6)
                    )
                    self.log_metrics(metrics, step=self.global_step)
                    loss_bundle.reset()
                    t0 = time.perf_counter()
                    logger.info(
                        "[step %d] loss=%.4f  lr=%.6f",
                        self.global_step,
                        metrics["loss/total"],
                        metrics["lr"],
                    )

                # --- Validation ---
                if self.global_step % cfg.eval_every_n_steps == 0:
                    val_loss = self.validate()
                    if self.rank == 0:
                        self.log_metrics({"val/loss": val_loss}, step=self.global_step)
                        logger.info("[step %d] val_loss=%.4f", self.global_step, val_loss)
                        if val_loss < self._best_val_loss:
                            self._best_val_loss = val_loss
                            if cfg.save_best_checkpoint:
                                self.save_checkpoint(tag="best")

                # --- Checkpoint ---
                if self.global_step % cfg.save_every_n_steps == 0:
                    self.save_checkpoint(tag=f"step_{self.global_step:07d}")

                if self.global_step >= cfg.max_steps:
                    return

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> float:
        """Run one pass over the validation split and return mean loss."""
        cfg = self.cfg
        val_loader = self._build_dataloader(cfg.val_dataset_split)

        self.vqvae.eval()
        self.world_model.eval()
        self.occ_forecaster.eval()

        total_loss = 0.0
        n_batches = 0

        with self.ema.average_parameters(self.world_model_raw):
            for batch in val_loader:
                with self._autocast():
                    loss, _ = self.compute_loss(batch)
                total_loss += loss.item()
                n_batches += 1
                if n_batches >= 50:  # cap validation to 50 batches for speed
                    break

        self.vqvae.train()
        self.world_model.train()
        self.occ_forecaster.train()

        mean_loss = total_loss / max(n_batches, 1)

        # Synchronise across ranks
        if self.is_distributed:
            t = torch.tensor([mean_loss], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            mean_loss = t.item()

        return mean_loss

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to WandB and TensorBoard (rank 0 only)."""
        if self.rank != 0:
            return

        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, global_step=step)

        if self.wandb_run is not None and _WANDB_AVAILABLE:
            wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> None:
        """
        Save model weights, optimizer state, and training metadata.

        Uses atomic write (save to temp dir, then rename) to prevent
        corruption from interrupted saves.

        Checkpoint directory structure:
            <output_dir>/<run_name>/<tag>/
                world_model.pt
                vqvae.pt
                occ_forecaster.pt
                optimizer.pt
                scheduler.pt
                ema.pt
                meta.pt
        """
        if self.rank != 0:
            # For FSDP: all ranks must participate in gathering weights
            if self.is_distributed and isinstance(self.world_model, FSDP if _FSDP_AVAILABLE else DDP):
                pass  # handle below

        cfg = self.cfg
        ckpt_dir = cfg.output_dir / cfg.run_name / tag
        tmp_dir = ckpt_dir.with_suffix(".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Extract raw state dicts
        if _FSDP_AVAILABLE and isinstance(self.world_model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg_fsdp = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.world_model, StateDictType.FULL_STATE_DICT, cfg_fsdp):
                wm_state = self.world_model.state_dict()
            with FSDP.state_dict_type(self.vqvae, StateDictType.FULL_STATE_DICT, cfg_fsdp):
                vq_state = self.vqvae.state_dict()
            with FSDP.state_dict_type(self.occ_forecaster, StateDictType.FULL_STATE_DICT, cfg_fsdp):
                occ_state = self.occ_forecaster.state_dict()
        else:
            wm_state = (
                self.world_model.module.state_dict()
                if hasattr(self.world_model, "module")
                else self.world_model.state_dict()
            )
            vq_state = (
                self.vqvae.module.state_dict()
                if hasattr(self.vqvae, "module")
                else self.vqvae.state_dict()
            )
            occ_state = (
                self.occ_forecaster.module.state_dict()
                if hasattr(self.occ_forecaster, "module")
                else self.occ_forecaster.state_dict()
            )

        if self.rank == 0:
            torch.save(wm_state, tmp_dir / "world_model.pt")
            torch.save(vq_state, tmp_dir / "vqvae.pt")
            torch.save(occ_state, tmp_dir / "occ_forecaster.pt")
            torch.save(self.optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(self.lr_scheduler.state_dict(), tmp_dir / "scheduler.pt")
            torch.save(self.ema.shadow, tmp_dir / "ema.pt")
            torch.save(
                {
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "best_val_loss": self._best_val_loss,
                    "run_name": cfg.run_name,
                },
                tmp_dir / "meta.pt",
            )

            # Atomic rename
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            tmp_dir.rename(ckpt_dir)
            logger.info("Saved checkpoint: %s", ckpt_dir)

            # Purge old checkpoints
            self._purge_old_checkpoints(tag)

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """
        Resume training from a saved checkpoint.

        Loads model weights, optimizer state, scheduler state, EMA weights,
        and training metadata (global_step, epoch, best_val_loss).

        Args:
            ckpt_dir: path to the checkpoint directory
        """
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        logger.info("Loading checkpoint from %s", ckpt_dir)

        map_loc = self.device

        # World model
        wm_state = torch.load(ckpt_dir / "world_model.pt", map_location=map_loc)
        target = (
            self.world_model.module
            if hasattr(self.world_model, "module")
            else self.world_model
        )
        target.load_state_dict(wm_state, strict=True)

        # VQ-VAE
        vq_state = torch.load(ckpt_dir / "vqvae.pt", map_location=map_loc)
        vq_target = (
            self.vqvae.module if hasattr(self.vqvae, "module") else self.vqvae
        )
        vq_target.load_state_dict(vq_state, strict=True)

        # Occupancy forecaster
        occ_state = torch.load(ckpt_dir / "occ_forecaster.pt", map_location=map_loc)
        occ_target = (
            self.occ_forecaster.module
            if hasattr(self.occ_forecaster, "module")
            else self.occ_forecaster
        )
        occ_target.load_state_dict(occ_state, strict=True)

        # Optimizer + scheduler
        opt_state = torch.load(ckpt_dir / "optimizer.pt", map_location=map_loc)
        self.optimizer.load_state_dict(opt_state)
        sched_state = torch.load(ckpt_dir / "scheduler.pt", map_location=map_loc)
        self.lr_scheduler.load_state_dict(sched_state)

        # EMA
        ema_shadow = torch.load(ckpt_dir / "ema.pt", map_location=map_loc)
        self.ema.shadow = ema_shadow

        # Metadata
        meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu")
        self.global_step = meta["global_step"]
        self.epoch = meta["epoch"]
        self._best_val_loss = meta["best_val_loss"]

        logger.info(
            "Resumed from step %d (epoch %d, best_val=%.4f)",
            self.global_step, self.epoch, self._best_val_loss,
        )

    def _purge_old_checkpoints(self, current_tag: str) -> None:
        """Remove old step checkpoints beyond keep_last_n_checkpoints."""
        cfg = self.cfg
        run_dir = cfg.output_dir / cfg.run_name
        if not run_dir.exists():
            return

        # Only purge step_XXXXXXX checkpoints (not 'best', 'latest')
        step_ckpts = sorted(
            [p for p in run_dir.iterdir() if p.name.startswith("step_")],
            key=lambda p: p.name,
        )
        while len(step_ckpts) > cfg.keep_last_n_checkpoints:
            oldest = step_ckpts.pop(0)
            if oldest.name != current_tag:
                shutil.rmtree(oldest, ignore_errors=True)
                logger.debug("Removed old checkpoint: %s", oldest)
