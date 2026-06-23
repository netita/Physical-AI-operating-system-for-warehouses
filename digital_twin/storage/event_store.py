"""
event_store.py — Persistent event store backed by PostgreSQL + TimescaleDB.

TimescaleDB is a time-series extension for PostgreSQL that provides:
  - automatic partitioning (hypertables) by time column
  - efficient time-range queries

Schema (created on first connection if not present)
---------------------------------------------------
CREATE TABLE warehouse_events (
    id          BIGSERIAL,
    event_time  TIMESTAMPTZ NOT NULL,          ← TimescaleDB time dimension
    event_type  TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    agent_ids   TEXT[],
    location_x  DOUBLE PRECISION,
    location_y  DOUBLE PRECISION,
    payload     JSONB,
    warehouse_id TEXT NOT NULL DEFAULT 'default'
);
SELECT create_hypertable('warehouse_events', 'event_time',
    if_not_exists => TRUE, migrate_data => TRUE);

Dependencies
------------
asyncpg — async PostgreSQL driver (already in pyproject.toml)
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain dataclass
# ---------------------------------------------------------------------------


@dataclass
class WarehouseEvent:
    """
    A discrete warehouse event to be persisted.

    Parameters
    ----------
    event_type : str
        E.g. "near_miss", "zone_violation", "pick_complete", "forklift_idle".
    severity : str
        "debug" | "info" | "warning" | "error" | "critical".
    agent_ids : list[str]
        Track IDs involved in the event.
    location_x, location_y : float
        World-frame coordinates of the event.
    payload : dict
        Arbitrary JSON-serialisable extra data.
    event_time : datetime.datetime | None
        UTC event time; defaults to now if None.
    warehouse_id : str
        Multi-site identifier.
    event_id : str
        UUID string (auto-generated).
    """

    event_type: str
    severity: str = "info"
    agent_ids: list[str] = field(default_factory=list)
    location_x: float = 0.0
    location_y: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    event_time: datetime.datetime | None = None
    warehouse_id: str = "default"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.event_time is None:
            self.event_time = datetime.datetime.now(tz=datetime.timezone.utc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "agent_ids": self.agent_ids,
            "location_x": self.location_x,
            "location_y": self.location_y,
            "payload": self.payload,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "warehouse_id": self.warehouse_id,
        }


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS warehouse_events (
    id              BIGSERIAL,
    event_id        UUID NOT NULL DEFAULT gen_random_uuid(),
    event_time      TIMESTAMPTZ NOT NULL,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'info',
    agent_ids       TEXT[],
    location_x      DOUBLE PRECISION,
    location_y      DOUBLE PRECISION,
    payload         JSONB,
    warehouse_id    TEXT NOT NULL DEFAULT 'default'
);
"""

_CREATE_HYPERTABLE_SQL = """
SELECT create_hypertable(
    'warehouse_events', 'event_time',
    if_not_exists => TRUE,
    migrate_data => TRUE
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_we_event_type ON warehouse_events (event_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_we_warehouse ON warehouse_events (warehouse_id, event_time DESC);
"""

_INSERT_SQL = """
INSERT INTO warehouse_events
    (event_id, event_time, event_type, severity, agent_ids,
     location_x, location_y, payload, warehouse_id)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

_QUERY_SQL = """
SELECT event_id, event_time, event_type, severity, agent_ids,
       location_x, location_y, payload, warehouse_id
FROM   warehouse_events
WHERE  event_time >= $1
  AND  event_time <= $2
  {type_filter}
  AND  warehouse_id = $3
ORDER  BY event_time DESC
LIMIT  $4
"""


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------


class EventStore:
    """
    Async event store backed by PostgreSQL / TimescaleDB.

    Parameters
    ----------
    dsn : str
        asyncpg DSN, e.g.
        "postgresql://warehousegpt:secret@localhost:5432/warehouse"
    warehouse_id : str
        Site identifier stored in every row.
    pool_min_size : int
        Minimum connections in the asyncpg pool.
    pool_max_size : int
        Maximum connections in the asyncpg pool.

    Usage
    -----
    store = EventStore("postgresql://user:pass@localhost/db")
    await store.connect()

    event = WarehouseEvent(event_type="near_miss", severity="warning", ...)
    await store.write_event(event)

    events = await store.query_events(start, end, ["near_miss", "collision"])
    await store.close()
    """

    def __init__(
        self,
        dsn: str,
        warehouse_id: str = "default",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._warehouse_id = warehouse_id
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        await self._ensure_schema()
        logger.info(
            "EventStore connected (pool=%d–%d, warehouse=%s)",
            self._pool_min, self._pool_max, self._warehouse_id,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("EventStore disconnected.")

    async def _ensure_schema(self) -> None:
        """Create table, hypertable, and indexes if they do not exist."""
        assert self._pool is not None, "Not connected"
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)

        # TimescaleDB hypertable — separate connection so a failure doesn't
        # abort the transaction that creates the table and indexes.
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(_CREATE_HYPERTABLE_SQL)
            except (asyncpg.UndefinedFunctionError, asyncpg.InvalidSchemaNameError):
                logger.warning(
                    "create_hypertable() not available — "
                    "is the TimescaleDB extension installed?"
                )

        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_INDEX_SQL)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write_event(self, event: WarehouseEvent) -> None:
        """
        Persist a single WarehouseEvent.

        Parameters
        ----------
        event : WarehouseEvent
            The event to store.
        """
        if self._pool is None:
            raise RuntimeError("EventStore not connected. Call await connect() first.")

        assert event.event_time is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                _INSERT_SQL,
                uuid.UUID(event.event_id),
                event.event_time,
                event.event_type,
                event.severity,
                event.agent_ids,
                event.location_x,
                event.location_y,
                json.dumps(event.payload),
                event.warehouse_id or self._warehouse_id,
            )
        logger.debug("Wrote event %s (%s)", event.event_id, event.event_type)

    async def write_events_bulk(self, events: list[WarehouseEvent]) -> None:
        """Batch-write multiple events in a single transaction."""
        if not events:
            return
        if self._pool is None:
            raise RuntimeError("EventStore not connected.")

        rows = [
            (
                uuid.UUID(e.event_id),
                e.event_time,
                e.event_type,
                e.severity,
                e.agent_ids,
                e.location_x,
                e.location_y,
                json.dumps(e.payload),
                e.warehouse_id or self._warehouse_id,
            )
            for e in events
            if e.event_time is not None
        ]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(_INSERT_SQL, rows)

        logger.debug("Bulk-wrote %d events.", len(rows))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query_events(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        event_types: list[str] | None = None,
        warehouse_id: str | None = None,
        limit: int = 1000,
    ) -> list[WarehouseEvent]:
        """
        Retrieve events within a time range.

        Parameters
        ----------
        start, end : datetime.datetime
            UTC time bounds (inclusive).
        event_types : list[str] | None
            Filter by event type(s).  None = all types.
        warehouse_id : str | None
            Override the default warehouse ID filter.
        limit : int
            Maximum number of rows returned.

        Returns
        -------
        list[WarehouseEvent]
            Events ordered by event_time DESC.
        """
        if self._pool is None:
            raise RuntimeError("EventStore not connected.")

        wid = warehouse_id or self._warehouse_id

        if event_types:
            type_filter = "AND event_type = ANY($5::text[])"
            query = _QUERY_SQL.format(type_filter=type_filter).replace("$4", "$6")
            # Rebuild query with correct param positions
            sql = """
SELECT event_id, event_time, event_type, severity, agent_ids,
       location_x, location_y, payload, warehouse_id
FROM   warehouse_events
WHERE  event_time >= $1
  AND  event_time <= $2
  AND  warehouse_id = $3
  AND  event_type = ANY($4::text[])
ORDER  BY event_time DESC
LIMIT  $5
"""
            args = (start, end, wid, event_types, limit)
        else:
            sql = """
SELECT event_id, event_time, event_type, severity, agent_ids,
       location_x, location_y, payload, warehouse_id
FROM   warehouse_events
WHERE  event_time >= $1
  AND  event_time <= $2
  AND  warehouse_id = $3
ORDER  BY event_time DESC
LIMIT  $4
"""
            args = (start, end, wid, limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)

        return [self._row_to_event(r) for r in rows]

    async def query_latest(
        self,
        n: int = 100,
        event_types: list[str] | None = None,
        warehouse_id: str | None = None,
    ) -> list[WarehouseEvent]:
        """Return the N most recent events."""
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        far_past = now - datetime.timedelta(days=365)
        return await self.query_events(
            start=far_past,
            end=now,
            event_types=event_types,
            warehouse_id=warehouse_id,
            limit=n,
        )

    async def count_events(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        event_type: str | None = None,
        warehouse_id: str | None = None,
    ) -> int:
        """Count events in a time window (fast aggregate via TimescaleDB)."""
        if self._pool is None:
            raise RuntimeError("EventStore not connected.")
        wid = warehouse_id or self._warehouse_id

        if event_type:
            sql = """
SELECT COUNT(*) FROM warehouse_events
WHERE event_time BETWEEN $1 AND $2
  AND warehouse_id = $3
  AND event_type = $4
"""
            args = (start, end, wid, event_type)
        else:
            sql = """
SELECT COUNT(*) FROM warehouse_events
WHERE event_time BETWEEN $1 AND $2
  AND warehouse_id = $3
"""
            args = (start, end, wid)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        return int(row["count"]) if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_event(row: asyncpg.Record) -> WarehouseEvent:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        elif payload is None:
            payload = {}

        return WarehouseEvent(
            event_id=str(row["event_id"]),
            event_time=row["event_time"],
            event_type=row["event_type"],
            severity=row["severity"],
            agent_ids=list(row["agent_ids"] or []),
            location_x=float(row["location_x"] or 0.0),
            location_y=float(row["location_y"] or 0.0),
            payload=payload,
            warehouse_id=row["warehouse_id"],
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "EventStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
