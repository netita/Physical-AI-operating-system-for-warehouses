"""
warehousegpt.synthetic_data
===========================
Synthetic data pipeline for Physical AI warehouse operations.

Provides end-to-end loading, preprocessing, augmentation, and batching
of the NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes
dataset together with supporting utilities.

Public surface
--------------
    WarehouseDatasetLoader  – HuggingFace hub download + annotation parsing
    VideoPreprocessor       – frame normalisation, resize, depth alignment
    WarehouseAugmentation   – photometric + geometric + weather augmentation
    DataPipeline            – full train / eval DataLoader construction
    DatasetStats            – statistics, visualisation, annotation validation
"""

from __future__ import annotations

from warehousegpt.synthetic_data.augmentation import WarehouseAugmentation
from warehousegpt.synthetic_data.data_pipeline import DataPipeline
from warehousegpt.synthetic_data.dataset_loader import WarehouseDatasetLoader
from warehousegpt.synthetic_data.preprocessing import VideoPreprocessor
from warehousegpt.synthetic_data.stats import DatasetStats

__all__: list[str] = [
    "WarehouseDatasetLoader",
    "VideoPreprocessor",
    "WarehouseAugmentation",
    "DataPipeline",
    "DatasetStats",
]
