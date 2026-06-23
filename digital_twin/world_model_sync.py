"""
world_model_sync.py — World Model Synchronizer.

Maintains a rolling buffer of past N WarehouseState snapshots and
periodically calls the world model to predict future states.  The
predicted states are blended with the observed states using a configurable
alpha (0 = pure observation, 1 = pure prediction).

The synchronizer runs as an async loop at a configurable rate (default 10 Hz).

Architecture
------------
                  ┌──────────────────┐
  camera frames   │                  │   predicted states
  ──────────────► │  WarehouseState  │ ──────────────────►
  sensor detects  │   Estimator      │
                  └─────────┬────────┘
                            │ state snapshots
                            ▼
                  ┌──────────────────┐
                  │  Rolling Buffer  │  (N frames)
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  World Model     │  trajectory / occupancy predictions
                  │  (async call)    │
                  └─────────┬────────┘
                            │ predicted state
                            ▼
                  ┌──────────────────┐
                  │  Blender         │  α * pred + (1-α) * obs → blended state
                  └──────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import numpy as np

from warehousegpt.digital_twin.state_estimation.warehouse_state import (
    AgentPose,
    WarehouseState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# World-model protocol (duck typing — no hard dependency on world_model pkg)
# ---------------------------------------------------------------------------


WorldModelCallable = Callable[
    [list[WarehouseState]],
    Coroutine[Any, Any, WarehouseState | None],
]
"""
Type alias for the async callable that the WorldModelSynchronizer uses to
request future-state predictions.

Signature::

    async def predict(history: list[WarehouseState]) -> WarehouseState | None

If the world model is unavailable or returns None, the synchronizer falls
back to the last observed state.
"""


# ---------------------------------------------------------------------------
# Default world-model stub (linear extrapolation of agent positions)
# ---------------------------------------------------------------------------


async def _linear_extrapolation_stub(
    history: list[WarehouseState],
) -> WarehouseState | None:
    """
    Naive stub: linearly extrapolate agent positions using the two most recent
    states.  Used when no real world model is wired in.
    """
    if len(history) < 2:
        return history[-1] if history else None

    s0, s1 = history[-2], history[-1]
    dt = s1.timestamp - s0.timestamp
    if dt <= 0:
        return s1

    def _extrapolate(
        poses0: list[AgentPose], poses1: list[AgentPose]
    ) -> list[AgentPose]:
        id_to_p0 = {p.agent_id: p for p in poses0}
        result: list[AgentPose] = []
        for p1 in poses1:
            p0 = id_to_p0.get(p1.agent_id)
            if p0 is None:
                result.append(p1)
                continue
            # One-step-ahead linear extrapolation
            result.append(
                AgentPose(
                    agent_id=p1.agent_id,
                    agent_type=p1.agent_type,
                    x=p1.x + p1.vx * dt,
                    y=p1.y + p1.vy * dt,
                    z=p1.z + p1.vz * dt,
                    heading_rad=p1.heading_rad,
                    vx=p1.vx,
                    vy=p1.vy,
                    vz=p1.vz,
                    confidence=p1.confidence * 0.9,  # decay confidence
                )
            )
        return result

    predicted = WarehouseState(
        timestamp=s1.timestamp + dt,
        frame_index=s1.frame_index + 1,
        forklift_poses=_extrapolate(s0.forklift_poses, s1.forklift_poses),
        worker_poses=_extrapolate(s0.worker_poses, s1.worker_poses),
        amr_poses=_extrapolate(s0.amr_poses, s1.amr_poses),
        inventory_locations=s1.inventory_locations,
        occupancy_grid=s1.occupancy_grid.copy(),
        active_incidents=s1.active_incidents,
        track_count=s1.track_count,
    )
    return predicted


# ---------------------------------------------------------------------------
# Blending helpers
# ---------------------------------------------------------------------------


def _blend_agent_poses(
    observed: list[AgentPose],
    predicted: list[AgentPose],
    alpha: float,
) -> list[AgentPose]:
    """
    Blend observed and predicted agent poses.

    alpha = 0.0 → pure observation
    alpha = 1.0 → pure prediction
    """
    if alpha <= 0.0:
        return observed
    if alpha >= 1.0:
        return predicted

    pred_map = {p.agent_id: p for p in predicted}
    blended: list[AgentPose] = []

    for obs in observed:
        pred = pred_map.get(obs.agent_id)
        if pred is None:
            blended.append(obs)
            continue

        blended.append(
            AgentPose(
                agent_id=obs.agent_id,
                agent_type=obs.agent_type,
                x=obs.x * (1 - alpha) + pred.x * alpha,
                y=obs.y * (1 - alpha) + pred.y * alpha,
                z=obs.z * (1 - alpha) + pred.z * alpha,
                heading_rad=obs.heading_rad,   # do not blend angle
                vx=obs.vx * (1 - alpha) + pred.vx * alpha,
                vy=obs.vy * (1 - alpha) + pred.vy * alpha,
                vz=obs.vz * (1 - alpha) + pred.vz * alpha,
                confidence=min(obs.confidence, pred.confidence),
            )
        )

    # Add agents only in prediction (not yet observed)
    obs_ids = {p.agent_id for p in observed}
    for pred in predicted:
        if pred.agent_id not in obs_ids:
            blended.append(pred)

    return blended


def _blend_states(
    observed: WarehouseState,
    predicted: WarehouseState,
    alpha: float,
) -> WarehouseState:
    """Create a blended WarehouseState from observed and predicted states."""
    return WarehouseState(
        timestamp=observed.timestamp,
        frame_index=observed.frame_index,
        forklift_poses=_blend_agent_poses(
            observed.forklift_poses, predicted.forklift_poses, alpha
        ),
        worker_poses=_blend_agent_poses(
            observed.worker_poses, predicted.worker_poses, alpha
        ),
        amr_poses=_blend_agent_poses(
            observed.amr_poses, predicted.amr_poses, alpha
        ),
        inventory_locations=observed.inventory_locations,
        occupancy_grid=observed.occupancy_grid,
        active_incidents=observed.active_incidents,
        track_count=observed.track_count,
    )


# ---------------------------------------------------------------------------
# WorldModelSynchronizer
# ---------------------------------------------------------------------------


@dataclass
class SyncMetrics:
    """Runtime metrics for monitoring."""

    total_frames: int = 0
    world_model_calls: int = 0
    world_model_failures: int = 0
    avg_loop_latency_ms: float = 0.0
    last_prediction_age_s: float = 0.0


class WorldModelSynchronizer:
    """
    Maintains a rolling buffer of past states, calls the world model for
    predictions, blends them with observations, and emits blended states at a
    fixed rate.

    Parameters
    ----------
    state_source : AsyncGenerator[WarehouseState, None] | None
        Async generator that yields fresh WarehouseState snapshots.
        Can be wired later via ``set_state_source()``.
    world_model_fn : WorldModelCallable | None
        Async callable for predictions.  Defaults to linear extrapolation stub.
    buffer_size : int
        Number of past states to keep in the rolling buffer.
    prediction_alpha : float
        Blending weight for the predicted state [0, 1].
    loop_hz : float
        Rate of the synchronizer's main async loop (default 10 Hz).
    """

    def __init__(
        self,
        state_source: "asyncio.Queue[WarehouseState] | None" = None,
        world_model_fn: WorldModelCallable | None = None,
        buffer_size: int = 30,
        prediction_alpha: float = 0.2,
        loop_hz: float = 10.0,
    ) -> None:
        self._state_queue: asyncio.Queue[WarehouseState] = (
            state_source if state_source is not None else asyncio.Queue(maxsize=32)
        )
        self._world_model = world_model_fn or _linear_extrapolation_stub
        self._buffer: deque[WarehouseState] = deque(maxlen=buffer_size)
        self._alpha = prediction_alpha
        self._loop_interval = 1.0 / loop_hz

        self._blended_state: WarehouseState | None = None
        self._latest_prediction: WarehouseState | None = None
        self._last_prediction_time: float = 0.0

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._output_callbacks: list[Callable[[WarehouseState], Any]] = []
        self.metrics = SyncMetrics()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_state_source(self, q: "asyncio.Queue[WarehouseState]") -> None:
        """Wire in a queue that receives WarehouseState snapshots."""
        self._state_queue = q

    def set_world_model(self, fn: WorldModelCallable) -> None:
        """Replace the world model callable."""
        self._world_model = fn

    def on_blended_state(self, cb: Callable[[WarehouseState], Any]) -> None:
        """Register a callback invoked with each blended WarehouseState."""
        self._output_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main async loop at ``loop_hz``.

        1. Drain the state queue into the rolling buffer.
        2. Call the world model on the buffered history.
        3. Blend the latest observation with the prediction.
        4. Fire output callbacks.
        """
        self._running = True
        logger.info(
            "WorldModelSynchronizer started at %.1f Hz "
            "(α=%.2f, buffer=%d)",
            1.0 / self._loop_interval,
            self._alpha,
            self._buffer.maxlen,
        )

        while self._running:
            t_start = time.monotonic()

            # Drain all freshly queued states into the rolling buffer
            while not self._state_queue.empty():
                try:
                    state = self._state_queue.get_nowait()
                    self._buffer.append(state)
                    self.metrics.total_frames += 1
                except asyncio.QueueEmpty:
                    break

            if self._buffer:
                observed = self._buffer[-1]   # most recent observation

                # Call world model for prediction
                prediction = await self._call_world_model()

                # Blend
                if prediction is not None:
                    blended = _blend_states(observed, prediction, self._alpha)
                else:
                    blended = observed

                self._blended_state = blended

                # Notify subscribers
                for cb in self._output_callbacks:
                    try:
                        result = cb(blended)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.error("Output callback error: %s", exc)

            # Sleep for the remainder of the interval
            elapsed = time.monotonic() - t_start
            self.metrics.avg_loop_latency_ms = (
                self.metrics.avg_loop_latency_ms * 0.9 + elapsed * 1000 * 0.1
            )
            sleep_for = self._loop_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        logger.info("WorldModelSynchronizer stopped.")

    async def start(self) -> None:
        """Launch ``run()`` as a background asyncio task."""
        self._task = asyncio.create_task(self.run(), name="world-model-sync")

    async def stop(self) -> None:
        """Stop the background task gracefully."""
        self._running = False
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_blended_state(self) -> WarehouseState | None:
        """Return the most recent blended state (may be None if not yet run)."""
        return self._blended_state

    def get_history(self) -> list[WarehouseState]:
        """Return a copy of the rolling buffer."""
        return list(self._buffer)

    def push_state(self, state: WarehouseState) -> None:
        """Push a state directly (useful when not using a queue source)."""
        try:
            self._state_queue.put_nowait(state)
        except asyncio.QueueFull:
            # Drop the oldest queued item and retry
            try:
                self._state_queue.get_nowait()
                self._state_queue.put_nowait(state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # World model call with error handling
    # ------------------------------------------------------------------

    async def _call_world_model(self) -> WarehouseState | None:
        history = list(self._buffer)
        if not history:
            return None

        self.metrics.world_model_calls += 1
        try:
            prediction = await asyncio.wait_for(
                self._world_model(history), timeout=self._loop_interval * 0.8
            )
            self._latest_prediction = prediction
            self._last_prediction_time = time.monotonic()
            self.metrics.last_prediction_age_s = 0.0
            return prediction
        except TimeoutError:
            logger.warning("World model prediction timed out — using last prediction.")
            self.metrics.world_model_failures += 1
        except Exception as exc:
            logger.error("World model error: %s", exc)
            self.metrics.world_model_failures += 1

        # Update age of last successful prediction
        if self._latest_prediction is not None:
            self.metrics.last_prediction_age_s = (
                time.monotonic() - self._last_prediction_time
            )
        return self._latest_prediction

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "WorldModelSynchronizer":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
