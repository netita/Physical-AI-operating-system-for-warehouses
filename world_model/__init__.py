"""
WarehouseGPT World Model — Phase 2
===================================
Causal video transformer for warehouse state prediction, occupancy forecasting,
and trajectory prediction.  Architecture overview:

  Video clips (B, T, C, H, W)
        │
        ▼
  WarehouseVQVAE / LatentTokenizer
        │  discrete tokens  OR  continuous latents
        ▼
  CausalVideoTransformer  ← action conditioning (forklift v, fork height)
        │
        ├─▶ next-frame token logits
        ├─▶ risk score head
        ├─▶ OccupancyForecaster (UNet head)
        └─▶ TrajectoryPredictor (GMM output)
"""

from warehousegpt.world_model.tokenizer.vqvae import WarehouseVQVAE, VectorQuantizer
from warehousegpt.world_model.tokenizer.latent_tokenizer import LatentTokenizer
from warehousegpt.world_model.transformer.model import (
    WarehouseWorldModel,
    WarehouseTransformerConfig,
)
from warehousegpt.world_model.transformer.attention import (
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
    RotaryPositionEmbedding,
)
from warehousegpt.world_model.prediction.temporal import TemporalPredictor
from warehousegpt.world_model.prediction.occupancy import OccupancyForecaster
from warehousegpt.world_model.prediction.trajectory import TrajectoryPredictor
from warehousegpt.world_model.training.trainer import WorldModelTrainer
from warehousegpt.world_model.training.config import TrainingConfig
from warehousegpt.world_model.training.dataset import WarehouseVideoDataset

__all__ = [
    # Tokenizers
    "WarehouseVQVAE",
    "VectorQuantizer",
    "LatentTokenizer",
    # Transformer
    "WarehouseWorldModel",
    "WarehouseTransformerConfig",
    # Attention modules
    "MultiHeadSelfAttention",
    "MultiHeadCrossAttention",
    "RotaryPositionEmbedding",
    # Predictors
    "TemporalPredictor",
    "OccupancyForecaster",
    "TrajectoryPredictor",
    # Training
    "WorldModelTrainer",
    "TrainingConfig",
    "WarehouseVideoDataset",
]
