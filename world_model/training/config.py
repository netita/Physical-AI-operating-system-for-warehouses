"""
Training configuration for the WarehouseGPT World Model.

All hyper-parameters are documented.  The defaults target a production
run on 8 × H100 80 GB GPUs with FSDP (Fully-Sharded Data Parallel).

Usage::

    from warehousegpt.world_model.training.config import TrainingConfig

    cfg = TrainingConfig()
    # Override for a quick dev run:
    cfg_dev = TrainingConfig(
        max_steps=1000,
        batch_size_per_gpu=1,
        use_wandb=False,
        mixed_precision="bf16",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Master configuration dataclass for WorldModelTrainer.

    Grouped into logical sections for readability.
    """

    # =========================================================================
    # Paths & Identifiers
    # =========================================================================

    run_name: str = "warehouse-world-model-v1"
    """Unique name for this training run (used for wandb, checkpoint dirs)."""

    output_dir: Path = Path("./checkpoints/world_model")
    """Root directory where checkpoints and logs are saved."""

    dataset_name: str = "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes"
    """HuggingFace Hub dataset identifier."""

    dataset_split: str = "train"
    """Which dataset split to use for training."""

    val_dataset_split: str = "validation"
    """Dataset split for validation."""

    # =========================================================================
    # Model Architecture (references WarehouseTransformerConfig)
    # =========================================================================

    d_model: int = 2048
    """Transformer hidden dimension."""

    num_layers: int = 24
    """Number of transformer blocks."""

    num_heads: int = 16
    """Number of attention heads."""

    codebook_size: int = 8192
    """VQ-VAE vocabulary size — must match the pretrained tokenizer."""

    embedding_dim: int = 256
    """VQ-VAE embedding dimension — must match the pretrained tokenizer."""

    image_height: int = 256
    """Input frame height in pixels."""

    image_width: int = 256
    """Input frame width in pixels."""

    patch_size: int = 16
    """Spatial patch size (pixels) for the transformer patch embedding."""

    max_frames: int = 64
    """Maximum context window length in frames."""

    context_frames: int = 16
    """Number of conditioning frames per training clip."""

    future_frames: int = 8
    """Number of future frames to predict per training clip."""

    action_dim: int = 6
    """Raw action vector dimension: [vx, vy, omega, fork_h, fork_tilt, load_w]."""

    action_embed_dim: int = 256
    """Projected action embedding dimension."""

    # =========================================================================
    # VQ-VAE Tokenizer
    # =========================================================================

    vqvae_checkpoint: Optional[Path] = None
    """
    Path to a pretrained VQ-VAE checkpoint.
    If None, the VQ-VAE is trained jointly with the transformer (stage-1).
    """

    vqvae_base_channels: int = 128
    """VQ-VAE encoder/decoder base channel width."""

    vqvae_channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    """Channel multiplier per stage in the VQ-VAE."""

    vqvae_num_res_blocks: int = 2
    """Residual blocks per stage in the VQ-VAE."""

    vqvae_commitment_cost: float = 0.25
    """VQ-VAE commitment loss weight (beta)."""

    vqvae_ema_decay: float = 0.99
    """EMA decay for codebook updates."""

    # =========================================================================
    # Training Schedule
    # =========================================================================

    max_steps: int = 500_000
    """Total gradient update steps (not epochs — dataset is large)."""

    warmup_steps: int = 2_000
    """Linear LR warmup steps."""

    eval_every_n_steps: int = 1_000
    """Run validation every N gradient steps."""

    save_every_n_steps: int = 5_000
    """Save a checkpoint every N gradient steps."""

    log_every_n_steps: int = 50
    """Log metrics every N steps."""

    # =========================================================================
    # Optimizer
    # =========================================================================

    optimizer: str = "adamw"
    """Optimizer name: 'adamw' | 'lion' | 'adafactor'."""

    learning_rate: float = 3e-4
    """Peak learning rate."""

    lr_scheduler: str = "cosine"
    """LR schedule: 'cosine' | 'linear' | 'constant'."""

    weight_decay: float = 0.1
    """AdamW weight decay."""

    beta1: float = 0.9
    """AdamW beta_1."""

    beta2: float = 0.95
    """AdamW beta_2 (0.95 is better than 0.999 for large models)."""

    eps: float = 1e-8
    """AdamW epsilon."""

    grad_clip: float = 1.0
    """Global gradient norm clipping threshold."""

    # =========================================================================
    # Batch & Gradient Accumulation
    # =========================================================================

    batch_size_per_gpu: int = 2
    """
    Per-GPU micro-batch size.
    Effective batch size = batch_size_per_gpu × num_gpus × gradient_accumulation_steps
    """

    gradient_accumulation_steps: int = 8
    """
    Number of forward passes before one optimizer step.
    With 8 H100s: effective_bs = 2 × 8 × 8 = 128 clips.
    """

    # =========================================================================
    # Mixed Precision & Compilation
    # =========================================================================

    mixed_precision: str = "bf16"
    """
    Mixed precision dtype: 'no' | 'fp16' | 'bf16'.
    BF16 is strongly recommended for H100/A100 GPUs.
    """

    compile_model: bool = False
    """
    If True, calls torch.compile() on the model for ~20% speedup.
    Disable for debugging; enable for production training.
    """

    # =========================================================================
    # Distributed Training
    # =========================================================================

    distributed_backend: str = "fsdp"
    """
    Distributed strategy: 'ddp' | 'fsdp'.
    FSDP (Fully Sharded Data Parallel) is required for >1B models.
    """

    fsdp_sharding_strategy: str = "full_shard"
    """
    FSDP sharding: 'full_shard' | 'shard_grad_op' | 'no_shard'.
    'full_shard' = ZeRO-3 — maximum memory efficiency.
    """

    fsdp_cpu_offload: bool = False
    """
    Offload FSDP parameters to CPU (saves GPU memory at cost of speed).
    """

    num_workers: int = 8
    """DataLoader worker processes per GPU."""

    prefetch_factor: int = 2
    """DataLoader prefetch factor."""

    # =========================================================================
    # Loss Weights
    # =========================================================================

    loss_recon_weight: float = 1.0
    """Weight for VQ-VAE reconstruction loss."""

    loss_vq_weight: float = 1.0
    """Weight for VQ-VAE codebook commitment loss."""

    loss_next_token_weight: float = 1.0
    """Weight for next-frame token prediction cross-entropy loss."""

    loss_occupancy_weight: float = 0.5
    """Weight for occupancy grid prediction loss."""

    loss_trajectory_weight: float = 0.5
    """Weight for trajectory prediction NLL loss."""

    loss_risk_weight: float = 0.1
    """Weight for risk score binary cross-entropy loss."""

    # =========================================================================
    # Data Augmentation
    # =========================================================================

    speed_jitter_range: tuple[float, float] = (0.75, 1.5)
    """
    Temporal speed jitter range.
    Clips are sampled at a random FPS multiplier within this range.
    """

    frame_dropout_prob: float = 0.1
    """Probability of dropping an individual frame (set to 0 to disable)."""

    color_jitter: bool = True
    """Apply random brightness/contrast/saturation jitter to RGB frames."""

    horizontal_flip: bool = False
    """
    Do NOT flip: warehouse layouts are asymmetric (aisle directions matter).
    """

    # =========================================================================
    # Validation
    # =========================================================================

    val_batch_size: int = 4
    """Batch size for validation (larger is fine since no grad tracking)."""

    val_context_frames: int = 16
    """Context frames for validation generation."""

    val_future_frames: int = 8
    """Future frames to generate for FVD / SSIM evaluation."""

    # =========================================================================
    # Logging & Monitoring
    # =========================================================================

    use_wandb: bool = True
    """Log to Weights & Biases."""

    wandb_project: str = "warehousegpt-world-model"
    """WandB project name."""

    wandb_entity: Optional[str] = None
    """WandB team/entity (None = personal account)."""

    use_tensorboard: bool = True
    """Log scalars to TensorBoard in addition to WandB."""

    tensorboard_dir: Path = Path("./runs/world_model")
    """TensorBoard log directory."""

    log_video_every_n_steps: int = 5_000
    """Log generated video comparisons to WandB every N steps."""

    # =========================================================================
    # Checkpointing
    # =========================================================================

    resume_from_checkpoint: Optional[Path] = None
    """
    Path to a checkpoint directory to resume from.
    If None, training starts from scratch.
    """

    keep_last_n_checkpoints: int = 3
    """Maximum number of checkpoint directories to keep."""

    save_best_checkpoint: bool = True
    """Always keep the checkpoint with the best validation loss."""

    # =========================================================================
    # Hardware / Reproducibility
    # =========================================================================

    seed: int = 42
    """Random seed for reproducibility."""

    device: str = "cuda"
    """Target device: 'cuda' | 'cpu'."""

    dtype_str: str = "bfloat16"
    """Torch dtype string for parameter storage: 'float32' | 'bfloat16'."""

    # =========================================================================
    # Derived properties
    # =========================================================================

    @property
    def clip_frames(self) -> int:
        """Total frames per training clip (context + future)."""
        return self.context_frames + self.future_frames

    @property
    def tokens_per_frame(self) -> int:
        """Number of spatial patch tokens per frame."""
        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)

    @property
    def tokens_per_clip(self) -> int:
        """Total tokens per training clip."""
        return self.clip_frames * self.tokens_per_frame

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.tensorboard_dir = Path(self.tensorboard_dir)
        if self.vqvae_checkpoint is not None:
            self.vqvae_checkpoint = Path(self.vqvae_checkpoint)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)

        # Validate
        assert self.mixed_precision in ("no", "fp16", "bf16"), (
            f"Invalid mixed_precision: {self.mixed_precision}"
        )
        assert self.distributed_backend in ("ddp", "fsdp"), (
            f"Invalid distributed_backend: {self.distributed_backend}"
        )
        assert self.optimizer in ("adamw", "lion", "adafactor"), (
            f"Invalid optimizer: {self.optimizer}"
        )
        assert self.lr_scheduler in ("cosine", "linear", "constant"), (
            f"Invalid lr_scheduler: {self.lr_scheduler}"
        )
