"""
throughput.py — Warehouse KPI analytics.

All analytics are computed from a rolling in-memory window of WarehouseState
snapshots.  For historical queries, wire in an EventStore.

KPIs implemented
----------------
picks_per_hour          — Pallet touch events / elapsed hours
forklift_utilization    — % of time forklifts are moving (speed > threshold)
aisle_congestion_score  — Ratio of aisle cells occupied vs total aisle cells

All public methods return plain Python dicts (JSON-serialisable) so they
can be returned directly from the FastAPI endpoint.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from warehousegpt.digital_twin.state_estimation.warehouse_state import WarehouseState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ForkliftTimeEntry:
    """Per-frame stats for a single forklift track."""

    track_id: str
    timestamp: float
    speed_mps: float          # total speed = sqrt(vx²+vy²)
    is_moving: bool


@dataclass
class PickEvent:
    """A detected pallet-pick completion."""

    timestamp: float
    forklift_id: str
    pallet_id: str
    location_x: float
    location_y: float
    zone: str = "unknown"


@dataclass
class AisleSample:
    """One occupancy sample of the aisle grid."""

    timestamp: float
    congestion_ratio: float   # 0.0 (clear) … 1.0 (fully blocked)


# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------


@dataclass
class AnalyticsConfig:
    """Tuneable knobs for the analytics engine."""

    # Sliding window for all aggregates (seconds)
    window_seconds: float = 3600.0  # 1 hour

    # Minimum speed (m/s) to consider a forklift "moving"
    moving_speed_threshold_mps: float = 0.2

    # Occupancy grid cells classified as "aisle" if True
    # Provide a (grid_h, grid_w) boolean mask; None = entire grid
    aisle_mask: np.ndarray | None = None

    # Minimum approach distance (m) between forklift and pallet to count a pick
    pick_proximity_m: float = 2.0

    # Sampling rate from incoming states (skip frames to reduce compute)
    sample_every_n: int = 1


# ---------------------------------------------------------------------------
# Main analytics class
# ---------------------------------------------------------------------------


class ThroughputAnalytics:
    """
    Rolling-window warehouse KPI analytics.

    Parameters
    ----------
    config : AnalyticsConfig | None
        Tuneable configuration; uses defaults if None.

    Usage
    -----
    analytics = ThroughputAnalytics()
    analytics.ingest(warehouse_state)

    kpis = {
        **analytics.compute_picks_per_hour(),
        **analytics.compute_forklift_utilization(),
        **analytics.compute_aisle_congestion_score(),
    }
    """

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self._cfg = config or AnalyticsConfig()

        # Rolling windows (deques auto-expire old entries in _prune())
        self._forklift_entries: deque[ForkliftTimeEntry] = deque()
        self._pick_events: deque[PickEvent] = deque()
        self._aisle_samples: deque[AisleSample] = deque()

        # Track last-known pallet locations per pallet ID (for pick detection)
        self._last_pallet_positions: dict[str, tuple[float, float]] = {}
        # Tracks which forklift last "held" a pallet (proximity-based)
        self._pallet_assignee: dict[str, str] = {}

        self._frame_count = 0
        self._last_prune_time = time.monotonic()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, state: WarehouseState) -> None:
        """
        Incorporate a new WarehouseState into the rolling analytics window.

        Call this every time a new state is produced by the estimator.
        """
        self._frame_count += 1
        if self._frame_count % self._cfg.sample_every_n != 0:
            return

        now = state.timestamp

        # ---- Forklift motion stats ----
        for fork in state.forklift_poses:
            speed = float(np.sqrt(fork.vx ** 2 + fork.vy ** 2 + fork.vz ** 2))
            entry = ForkliftTimeEntry(
                track_id=fork.agent_id,
                timestamp=now,
                speed_mps=speed,
                is_moving=speed >= self._cfg.moving_speed_threshold_mps,
            )
            self._forklift_entries.append(entry)

        # ---- Pick detection (proximity heuristic) ----
        self._detect_picks(state)

        # ---- Aisle congestion ----
        if state.occupancy_grid is not None:
            congestion = self._compute_congestion(state.occupancy_grid)
            self._aisle_samples.append(AisleSample(timestamp=now, congestion_ratio=congestion))

        # Prune old data periodically (every 30 s)
        if now - self._last_prune_time > 30.0:
            self._prune(now)
            self._last_prune_time = now

    # ------------------------------------------------------------------
    # KPI: picks per hour
    # ------------------------------------------------------------------

    def compute_picks_per_hour(
        self,
        window_seconds: float | None = None,
    ) -> dict[str, Any]:
        """
        Estimate pallet picks per hour over a sliding window.

        Returns
        -------
        dict with keys:
            picks_in_window : int
            window_seconds  : float
            picks_per_hour  : float
            by_zone         : dict[str, float]  — picks/hr per zone
        """
        w = window_seconds or self._cfg.window_seconds
        cutoff = time.monotonic() - w

        recent = [p for p in self._pick_events if p.timestamp >= cutoff]
        hours = w / 3600.0

        by_zone: dict[str, int] = defaultdict(int)
        for pick in recent:
            by_zone[pick.zone] += 1

        return {
            "picks_in_window": len(recent),
            "window_seconds": w,
            "picks_per_hour": round(len(recent) / hours, 2) if hours > 0 else 0.0,
            "by_zone": {z: round(c / hours, 2) for z, c in by_zone.items()},
        }

    # ------------------------------------------------------------------
    # KPI: forklift utilization
    # ------------------------------------------------------------------

    def compute_forklift_utilization(
        self,
        window_seconds: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute per-forklift utilization as % of time moving.

        Returns
        -------
        dict with keys:
            fleet_utilization_pct : float  — fleet-wide average
            per_forklift          : dict[str, float]
            moving_speed_threshold: float
            window_seconds        : float
        """
        w = window_seconds or self._cfg.window_seconds
        cutoff = time.monotonic() - w

        # Group entries by track_id
        by_track: dict[str, list[ForkliftTimeEntry]] = defaultdict(list)
        for e in self._forklift_entries:
            if e.timestamp >= cutoff:
                by_track[e.track_id].append(e)

        per_fork: dict[str, float] = {}
        for tid, entries in by_track.items():
            if not entries:
                continue
            moving = sum(1 for e in entries if e.is_moving)
            per_fork[tid] = round(moving / len(entries) * 100.0, 2)

        fleet_avg = round(float(np.mean(list(per_fork.values()))), 2) if per_fork else 0.0

        return {
            "fleet_utilization_pct": fleet_avg,
            "per_forklift": per_fork,
            "moving_speed_threshold_mps": self._cfg.moving_speed_threshold_mps,
            "window_seconds": w,
            "active_forklifts": len(per_fork),
        }

    # ------------------------------------------------------------------
    # KPI: aisle congestion
    # ------------------------------------------------------------------

    def compute_aisle_congestion_score(
        self,
        window_seconds: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute average aisle congestion ratio over the window.

        Congestion ratio = (occupied aisle cells) / (total aisle cells).
        1.0 = fully blocked, 0.0 = completely free.

        Returns
        -------
        dict with keys:
            current_congestion  : float  — most recent sample
            avg_congestion      : float  — rolling average
            max_congestion      : float  — peak in window
            window_seconds      : float
            sample_count        : int
        """
        w = window_seconds or self._cfg.window_seconds
        cutoff = time.monotonic() - w

        recent = [s for s in self._aisle_samples if s.timestamp >= cutoff]
        if not recent:
            return {
                "current_congestion": 0.0,
                "avg_congestion": 0.0,
                "max_congestion": 0.0,
                "window_seconds": w,
                "sample_count": 0,
            }

        ratios = [s.congestion_ratio for s in recent]
        return {
            "current_congestion": round(recent[-1].congestion_ratio, 4),
            "avg_congestion": round(float(np.mean(ratios)), 4),
            "max_congestion": round(float(np.max(ratios)), 4),
            "window_seconds": w,
            "sample_count": len(recent),
        }

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def compute_all(
        self, window_seconds: float | None = None
    ) -> dict[str, Any]:
        """Return all KPIs in a single dict."""
        return {
            "picks": self.compute_picks_per_hour(window_seconds),
            "utilization": self.compute_forklift_utilization(window_seconds),
            "congestion": self.compute_aisle_congestion_score(window_seconds),
            "total_frames_ingested": self._frame_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_picks(self, state: WarehouseState) -> None:
        """
        Heuristic pick detection: if a pallet moves from its last location
        by more than `pick_proximity_m` and a forklift is nearby, record a pick.
        """
        now = state.timestamp
        thresh = self._cfg.pick_proximity_m

        # Build forklift lookup
        forks = {f.agent_id: f for f in state.forklift_poses}

        for inv in state.inventory_locations:
            pid = inv.item_id
            last_pos = self._last_pallet_positions.get(pid)

            if last_pos is not None:
                dx = inv.x - last_pos[0]
                dy = inv.y - last_pos[1]
                dist_moved = float(np.sqrt(dx ** 2 + dy ** 2))

                if dist_moved >= thresh:
                    # Pallet moved — check if any forklift is close
                    closest_fork_id: str | None = None
                    closest_dist = float("inf")
                    for fid, fork in forks.items():
                        d = float(np.sqrt((fork.x - inv.x) ** 2 + (fork.y - inv.y) ** 2))
                        if d < closest_dist:
                            closest_dist = d
                            closest_fork_id = fid

                    if closest_fork_id and closest_dist <= thresh * 2:
                        pick = PickEvent(
                            timestamp=now,
                            forklift_id=closest_fork_id,
                            pallet_id=pid,
                            location_x=inv.x,
                            location_y=inv.y,
                            zone=inv.zone,
                        )
                        self._pick_events.append(pick)
                        logger.debug(
                            "Pick event: forklift=%s pallet=%s zone=%s",
                            closest_fork_id, pid, inv.zone,
                        )

            self._last_pallet_positions[pid] = (inv.x, inv.y)

    def _compute_congestion(self, occupancy_grid: np.ndarray) -> float:
        """
        Return congestion ratio from the occupancy grid.

        If an aisle mask is configured, restrict to those cells.
        """
        if self._cfg.aisle_mask is not None:
            mask = self._cfg.aisle_mask
            if mask.shape != occupancy_grid.shape:
                # Shape mismatch — fall back to full grid
                total = occupancy_grid.size
                occupied = int(occupancy_grid.sum())
            else:
                total = int(mask.sum())
                occupied = int((occupancy_grid * mask).sum())
        else:
            total = occupancy_grid.size
            occupied = int(occupancy_grid.sum())

        if total == 0:
            return 0.0
        return float(occupied) / total

    def _prune(self, now: float) -> None:
        """Remove entries older than the window from all deques."""
        cutoff = now - self._cfg.window_seconds

        while self._forklift_entries and self._forklift_entries[0].timestamp < cutoff:
            self._forklift_entries.popleft()
        while self._pick_events and self._pick_events[0].timestamp < cutoff:
            self._pick_events.popleft()
        while self._aisle_samples and self._aisle_samples[0].timestamp < cutoff:
            self._aisle_samples.popleft()

    # ------------------------------------------------------------------
    # State inspection (useful for tests and dashboards)
    # ------------------------------------------------------------------

    @property
    def pick_count(self) -> int:
        """Total picks recorded in the rolling window."""
        cutoff = time.monotonic() - self._cfg.window_seconds
        return sum(1 for p in self._pick_events if p.timestamp >= cutoff)

    @property
    def active_forklift_count(self) -> int:
        """Number of unique forklifts seen in the current window."""
        cutoff = time.monotonic() - self._cfg.window_seconds
        ids = {e.track_id for e in self._forklift_entries if e.timestamp >= cutoff}
        return len(ids)
