"""Detector sub-package for the Safety AI module."""

from __future__ import annotations

from warehousegpt.safety_ai.detectors.collision import CollisionDetector, CollisionEvent
from warehousegpt.safety_ai.detectors.fire import FireDetector, FireEvent
from warehousegpt.safety_ai.detectors.near_miss import NearMissDetector, NearMissEvent
from warehousegpt.safety_ai.detectors.worker_safety import (
    ErgoRiskScore,
    FatigueIndicator,
    PPEStatus,
    WorkerSafetyMonitor,
)
from warehousegpt.safety_ai.detectors.zone_violation import (
    ViolationEvent,
    ZoneViolationDetector,
)

__all__: list[str] = [
    # Near-miss
    "NearMissDetector",
    "NearMissEvent",
    # Collision
    "CollisionDetector",
    "CollisionEvent",
    # Fire
    "FireDetector",
    "FireEvent",
    # Zone violation
    "ZoneViolationDetector",
    "ViolationEvent",
    # Worker safety
    "WorkerSafetyMonitor",
    "PPEStatus",
    "ErgoRiskScore",
    "FatigueIndicator",
]
