"""
state_store.py — Real-time warehouse state store backed by Redis.

Redis data model
----------------
Key: warehouse:<warehouse_id>:state
Type: Hash (via SET with JSON serialisation for simplicity)
TTL: configurable (default 60 s; rolling refresh on every write)

Pub/Sub channel: warehouse:<warehouse_id>:state_updates
  Message format: JSON-serialised WarehouseState (sans occupancy_grid)

Why not XADD (streams)?
  Redis Streams would be ideal for fan-out to multiple consumers, but for
  simplicity we use the classic PUBLISH/SUBSCRIBE pattern here.  The
  state_store can be extended with XADD later.

Dependencies
------------
redis[hiredis] — already in pyproject.toml
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import redis.asyncio as aioredis

from warehousegpt.digital_twin.state_estimation.warehouse_state import WarehouseState

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 60
_STATE_KEY_TEMPLATE = "warehouse:{warehouse_id}:state"
_CHANNEL_TEMPLATE = "warehouse:{warehouse_id}:state_updates"


class StateStore:
    """
    Redis-backed real-time state store.

    Usage
    -----
    store = StateStore("redis://localhost:6379", warehouse_id="site_A")
    await store.connect()

    # Writer side
    await store.set_state(warehouse_state)

    # Reader side — single latest value
    state = await store.get_latest_state()

    # Reader side — streaming updates
    async for state in store.subscribe_state_updates():
        process(state)

    await store.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        warehouse_id: str = "default",
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_history: int = 1000,
    ) -> None:
        self._redis_url = redis_url
        self._warehouse_id = warehouse_id
        self._ttl = ttl_seconds
        self._max_history = max_history

        self._state_key = _STATE_KEY_TEMPLATE.format(warehouse_id=warehouse_id)
        self._channel = _CHANNEL_TEMPLATE.format(warehouse_id=warehouse_id)
        self._history_key = f"warehouse:{warehouse_id}:state_history"

        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open async Redis connection."""
        self._client = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Verify connectivity
        await self._client.ping()
        logger.info(
            "StateStore connected to Redis (%s, warehouse=%s)",
            self._redis_url,
            self._warehouse_id,
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("StateStore disconnected from Redis.")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def set_state(self, state: WarehouseState) -> None:
        """
        Persist the current WarehouseState to Redis and publish an update.

        - Stores JSON under ``warehouse:<id>:state`` with a rolling TTL.
        - Publishes the JSON to ``warehouse:<id>:state_updates``.
        - Appends a trimmed history list for replay / analytics.

        Parameters
        ----------
        state : WarehouseState
            The state to store.  The occupancy_grid numpy array is omitted
            from the serialised form to keep payloads compact.
        """
        if self._client is None:
            raise RuntimeError("StateStore not connected. Call await connect() first.")

        payload = json.dumps(state.as_dict())

        pipe = self._client.pipeline()
        # Overwrite latest state with rolling TTL
        pipe.set(self._state_key, payload, ex=self._ttl)
        # Publish to subscribers
        pipe.publish(self._channel, payload)
        # Append to history list (trimmed to max_history)
        pipe.lpush(self._history_key, payload)
        pipe.ltrim(self._history_key, 0, self._max_history - 1)
        pipe.expire(self._history_key, self._ttl * 10)

        await pipe.execute()
        logger.debug("State written (frame=%d)", state.frame_index)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_latest_state(self) -> WarehouseState | None:
        """
        Return the most recently stored WarehouseState, or None if expired/absent.

        Note: the occupancy_grid field is not restored from Redis
        (it was never serialised); callers needing the grid should hold
        the WarehouseStateEstimator instance directly.
        """
        if self._client is None:
            raise RuntimeError("StateStore not connected.")

        raw = await self._client.get(self._state_key)
        if raw is None:
            return None

        return self._deserialise(raw)

    async def get_state_history(
        self, n: int = 100
    ) -> list[WarehouseState]:
        """
        Return the N most recent states from the Redis history list.
        Ordered newest → oldest.
        """
        if self._client is None:
            raise RuntimeError("StateStore not connected.")

        items = await self._client.lrange(self._history_key, 0, n - 1)
        states: list[WarehouseState] = []
        for raw in items:
            try:
                states.append(self._deserialise(raw))
            except Exception as exc:
                logger.warning("Failed to deserialise history entry: %s", exc)
        return states

    async def state_exists(self) -> bool:
        """Return True if a live state exists in Redis (not expired)."""
        if self._client is None:
            raise RuntimeError("StateStore not connected.")
        return bool(await self._client.exists(self._state_key))

    # ------------------------------------------------------------------
    # Pub/Sub streaming
    # ------------------------------------------------------------------

    async def subscribe_state_updates(
        self,
        timeout: float | None = None,
    ) -> AsyncGenerator[WarehouseState, None]:
        """
        Async generator that yields WarehouseState updates in real-time
        via Redis Pub/Sub.

        The generator never terminates on its own; callers should break
        out of the loop or use a timeout.

        Parameters
        ----------
        timeout : float | None
            Seconds to wait for the next message.  None = block forever.

        Yields
        ------
        WarehouseState
            Each freshly published state update.
        """
        if self._client is None:
            raise RuntimeError("StateStore not connected.")

        # Create a separate client for the blocking subscribe call
        sub_client = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        pubsub = sub_client.pubsub()
        await pubsub.subscribe(self._channel)
        logger.info("Subscribed to channel '%s'", self._channel)

        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                raw = message["data"]
                if not isinstance(raw, str):
                    continue
                try:
                    state = self._deserialise(raw)
                    yield state
                except Exception as exc:
                    logger.warning("Failed to parse pub/sub message: %s", exc)
        finally:
            await pubsub.unsubscribe(self._channel)
            await sub_client.aclose()
            logger.info("Unsubscribed from channel '%s'", self._channel)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialise(raw: str) -> WarehouseState:
        """Reconstruct a WarehouseState from its JSON representation."""
        from warehousegpt.digital_twin.state_estimation.warehouse_state import (
            AgentPose,
            Incident,
            InventoryLocation,
        )

        data: dict[str, Any] = json.loads(raw)

        def _agent(d: dict[str, Any], atype: str) -> AgentPose:
            import math
            return AgentPose(
                agent_id=d["agent_id"],
                agent_type=d.get("agent_type", atype),
                x=d.get("x", 0.0),
                y=d.get("y", 0.0),
                z=d.get("z", 0.0),
                heading_rad=math.radians(d.get("heading_deg", 0.0)),
                vx=d.get("vx", 0.0),
                vy=d.get("vy", 0.0),
                confidence=d.get("confidence", 1.0),
            )

        def _incident(d: dict[str, Any]) -> Incident:
            loc = d.get("location", {})
            return Incident(
                incident_id=d.get("incident_id", ""),
                incident_type=d.get("incident_type", "unknown"),
                severity=d.get("severity", "info"),
                timestamp=d.get("timestamp", 0.0),
                description=d.get("description", ""),
                agent_ids=d.get("agent_ids", []),
                location_x=loc.get("x", 0.0),
                location_y=loc.get("y", 0.0),
                resolved=d.get("resolved", False),
            )

        def _inventory(d: dict[str, Any]) -> InventoryLocation:
            return InventoryLocation(
                item_id=d.get("item_id", ""),
                barcode=d.get("barcode", ""),
                x=d.get("x", 0.0),
                y=d.get("y", 0.0),
                z=d.get("z", 0.0),
                zone=d.get("zone", "unknown"),
            )

        import numpy as np

        return WarehouseState(
            timestamp=data.get("timestamp", 0.0),
            frame_index=data.get("frame_index", 0),
            forklift_poses=[_agent(p, "forklift") for p in data.get("forklifts", [])],
            worker_poses=[_agent(p, "worker") for p in data.get("workers", [])],
            amr_poses=[_agent(p, "amr") for p in data.get("amrs", [])],
            inventory_locations=[_inventory(i) for i in data.get("inventory", [])],
            occupancy_grid=np.zeros((200, 200), dtype=np.uint8),  # not stored in Redis
            active_incidents=[_incident(i) for i in data.get("active_incidents", [])],
            track_count=data.get("track_count", 0),
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "StateStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
