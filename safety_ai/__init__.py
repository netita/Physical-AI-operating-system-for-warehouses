"""
warehousegpt.safety_ai
======================
Warehouse Safety AI subsystem — Phase 3 of WarehouseGPT.

Provides real-time detection and alerting for:
- Near-miss events between workers and forklifts
- Post-hoc collision analysis
- Fire / smoke detection (visual + thermal)
- Restricted-zone violations
- Worker PPE compliance, ergonomic risk, and fatigue monitoring

Sub-packages
------------
detectors   — Per-hazard detection modules
training    — Fine-tuning pipelines and evaluation metrics
labeling    — Labeling strategies and active-learning queries

Public re-exports keep top-level imports clean:

    from warehousegpt.safety_ai import (
        NearMissDetector, CollisionDetector, FireDetector,
        ZoneViolationDetector, WorkerSafetyMonitor,
    )
"""

from __future__ import annotations

from warehousegpt.safety_ai.detectors.collision import CollisionDetector
from warehousegpt.safety_ai.detectors.fire import FireDetector
from warehousegpt.safety_ai.detectors.near_miss import NearMissDetector
from warehousegpt.safety_ai.detectors.worker_safety import WorkerSafetyMonitor
from warehousegpt.safety_ai.detectors.zone_violation import ZoneViolationDetector

__all__: list[str] = [
    "NearMissDetector",
    "CollisionDetector",
    "FireDetector",
    "ZoneViolationDetector",
    "WorkerSafetyMonitor",
]
