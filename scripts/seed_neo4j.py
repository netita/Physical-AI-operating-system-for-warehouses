"""
scripts/seed_neo4j.py — Seed the Neo4j knowledge graph with real warehouse data.

Graph structure
---------------
(Zone)-[:CONTAINS]->(Aisle)-[:CONTAINS]->(Slot)-[:STORED_IN {count}]->(Zone)
(Robot)-[:LOCATED_IN]->(Zone)
(Regulation) standalone

Slot key format:  zone/aisle/rack/level/position
  e.g.  A/03/1/2/04  → Zone A, Aisle 03, Rack 1, Level 2, Position 04

Run
---
  cd warehousegpt
  source .venv/bin/activate
  NEO4J_PASSWORD=change_me_strong_neo4j_password python scripts/seed_neo4j.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "change_me_strong_neo4j_password")

SLOTS_FILE = Path(__file__).parent.parent / "user_data" / "info_slots.json"

ZONE_META = {
    "A": {"name": "Zone A — Fast-Pick",  "description": "High-velocity SKUs, 8 aisles, ambient temperature", "area_m2": 2500},
    "B": {"name": "Zone B — Bulk Storage","description": "Bulk and overstock items, 4 aisles, heavy racking", "area_m2": 5000},
    "C": {"name": "Zone C — Cold Storage","description": "Temperature-controlled zone, 4 aisles, 4°C", "area_m2": 1500},
}

ROBOTS = [
    {"id": "AMR-001", "type": "AMR",      "zone": "A", "status": "active",   "battery": 88, "model": "MiR200"},
    {"id": "AMR-002", "type": "AMR",      "zone": "B", "status": "active",   "battery": 71, "model": "MiR200"},
    {"id": "AMR-003", "type": "AMR",      "zone": "C", "status": "charging", "battery": 22, "model": "MiR200"},
    {"id": "AMR-004", "type": "AMR",      "zone": "A", "status": "active",   "battery": 95, "model": "MiR200"},
    {"id": "FORK-001","type": "Forklift", "zone": "B", "status": "picking",  "battery": 75, "model": "STILL RX60"},
    {"id": "FORK-002","type": "Forklift", "zone": "B", "status": "transporting","battery": 44, "model": "STILL RX60"},
    {"id": "FORK-003","type": "Forklift", "zone": "A", "status": "transporting","battery": 52, "model": "STILL RX60"},
    {"id": "FORK-004","type": "Forklift", "zone": "C", "status": "idle",     "battery": 91, "model": "STILL RX60"},
]

REGULATIONS = [
    {
        "id": "REG-OSHA-178",
        "source": "OSHA",
        "code": "1910.178(l)",
        "title": "Powered Industrial Truck Operator Training",
        "summary": "Requires operators to be trained and certified before operating powered industrial trucks.",
        "applies_to": ["Forklift", "AMR"],
    },
    {
        "id": "REG-ISO-3691",
        "source": "ISO",
        "code": "3691-4:2020",
        "title": "Driverless Industrial Trucks Safety Requirements",
        "summary": "Safety requirements and verification for driverless industrial trucks and their systems.",
        "applies_to": ["AMR"],
    },
    {
        "id": "REG-OSHA-1910",
        "source": "OSHA",
        "code": "1910.303",
        "title": "General Electrical Safety in the Workplace",
        "summary": "Electrical safety standards including charging stations for electric forklifts and AMRs.",
        "applies_to": ["Forklift", "AMR"],
    },
    {
        "id": "REG-ANSI-ITSDF",
        "source": "ANSI/ITSDF",
        "code": "B56.5-2019",
        "title": "Safety Standard for Driverless Automatic Guided Industrial Vehicles",
        "summary": "Safety requirements for automatic guided vehicles in industrial settings.",
        "applies_to": ["AMR"],
    },
]

BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"  {msg}")


def run_batch(session: Any, query: str, rows: list[dict[str, Any]], label: str) -> None:
    """Execute a batched UNWIND query and print timing."""
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        session.run(query, rows=chunk)
    elapsed = time.time() - t0
    log(f"{label}: {len(rows)} nodes  ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Schema constraints
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT zone_id IF NOT EXISTS FOR (z:Zone)       REQUIRE z.id IS UNIQUE",
    "CREATE CONSTRAINT aisle_id IF NOT EXISTS FOR (a:Aisle)     REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT slot_id IF NOT EXISTS FOR (s:Slot)       REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT robot_id IF NOT EXISTS FOR (r:Robot)     REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT reg_id IF NOT EXISTS FOR (r:Regulation)  REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (i:Incident) REQUIRE i.id IS UNIQUE",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed(driver: Any) -> None:
    slots_data: dict[str, list[str]] = json.loads(SLOTS_FILE.read_text())

    # ------------------------------------------------------------------ #
    # 1. Constraints & indexes
    # ------------------------------------------------------------------ #
    with driver.session() as s:
        for cypher in CONSTRAINTS:
            s.run(cypher)
    log("Constraints OK")

    # ------------------------------------------------------------------ #
    # 2. Zones
    # ------------------------------------------------------------------ #
    zone_rows = [
        {"id": z, **meta}
        for z, meta in ZONE_META.items()
    ]
    with driver.session() as s:
        run_batch(
            s,
            "UNWIND $rows AS r MERGE (z:Zone {id: r.id}) SET z += r",
            zone_rows,
            "Zones",
        )

    # ------------------------------------------------------------------ #
    # 3. Aisles  (zone/aisle_num)
    # ------------------------------------------------------------------ #
    aisles: dict[str, dict[str, Any]] = {}
    for slot_key in slots_data:
        parts = slot_key.split("/")
        zone_id   = parts[0]
        aisle_num = parts[1]
        aisle_id  = f"{zone_id}/{aisle_num}"
        if aisle_id not in aisles:
            aisles[aisle_id] = {
                "id": aisle_id,
                "zone_id": zone_id,
                "aisle_num": aisle_num,
                "name": f"Aisle {aisle_num}",
            }

    aisle_rows = list(aisles.values())
    with driver.session() as s:
        run_batch(
            s,
            """
            UNWIND $rows AS r
            MERGE (a:Aisle {id: r.id}) SET a += {name: r.name, aisle_num: r.aisle_num}
            WITH a, r
            MATCH (z:Zone {id: r.zone_id})
            MERGE (z)-[:CONTAINS]->(a)
            """,
            aisle_rows,
            "Aisles + Zone→Aisle",
        )

    # ------------------------------------------------------------------ #
    # 4. Slots  (with box_ids as property, box_count, occupancy)
    # ------------------------------------------------------------------ #
    slot_rows = []
    for slot_key, boxes in slots_data.items():
        parts     = slot_key.split("/")
        zone_id   = parts[0]
        aisle_num = parts[1]
        aisle_id  = f"{zone_id}/{aisle_num}"
        slot_rows.append({
            "id":        slot_key,
            "aisle_id":  aisle_id,
            "zone_id":   zone_id,
            "rack":      int(parts[2]),
            "level":     int(parts[3]),
            "position":  parts[4],
            "box_ids":   boxes,
            "box_count": len(boxes),
            "capacity":  10,
            "occupancy": round(len(boxes) / 10, 2),
        })

    with driver.session() as s:
        run_batch(
            s,
            """
            UNWIND $rows AS r
            MERGE (s:Slot {id: r.id})
            SET s += {rack: r.rack, level: r.level, position: r.position,
                      box_ids: r.box_ids, box_count: r.box_count,
                      capacity: r.capacity, occupancy: r.occupancy,
                      zone_id: r.zone_id}
            WITH s, r
            MATCH (a:Aisle {id: r.aisle_id})
            MERGE (a)-[:CONTAINS]->(s)
            """,
            slot_rows,
            "Slots + Aisle→Slot",
        )

    # ------------------------------------------------------------------ #
    # 5. Robots
    # ------------------------------------------------------------------ #
    with driver.session() as s:
        run_batch(
            s,
            """
            UNWIND $rows AS r
            MERGE (robot:Robot {id: r.id})
            SET robot += {type: r.type, status: r.status,
                          battery: r.battery, model: r.model}
            WITH robot, r
            MATCH (z:Zone {id: r.zone})
            MERGE (robot)-[:LOCATED_IN]->(z)
            """,
            ROBOTS,
            "Robots + Robot→Zone",
        )

    # ------------------------------------------------------------------ #
    # 6. Regulations
    # ------------------------------------------------------------------ #
    with driver.session() as s:
        run_batch(
            s,
            "UNWIND $rows AS r MERGE (reg:Regulation {id: r.id}) SET reg += r",
            REGULATIONS,
            "Regulations",
        )

    # ------------------------------------------------------------------ #
    # 7. Regulation → Zone  REQUIRED_BY relationships
    # ------------------------------------------------------------------ #
    # All regulations apply to all zones; OSHA-178 also applies to C (forklifts visit)
    reg_zone_pairs = [
        {"reg_id": reg["id"], "zone_id": z}
        for reg in REGULATIONS
        for z in ZONE_META
    ]
    with driver.session() as s:
        run_batch(
            s,
            """
            UNWIND $rows AS r
            MATCH (reg:Regulation {id: r.reg_id}), (z:Zone {id: r.zone_id})
            MERGE (reg)-[:REQUIRED_BY]->(z)
            """,
            reg_zone_pairs,
            "Regulation→Zone REQUIRED_BY",
        )

    # ------------------------------------------------------------------ #
    # 8. Summary
    # ------------------------------------------------------------------ #
    with driver.session() as s:
        counts = s.run("""
            RETURN
              count{ MATCH (z:Zone)       RETURN z } AS zones,
              count{ MATCH (a:Aisle)      RETURN a } AS aisles,
              count{ MATCH (s:Slot)       RETURN s } AS slots,
              count{ MATCH (r:Robot)      RETURN r } AS robots,
              count{ MATCH (r:Regulation) RETURN r } AS regulations,
              count{ MATCH ()-[rel]->()   RETURN rel } AS relationships
        """).single()
        print()
        print("  ┌─────────────────────────────┐")
        print(f"  │  Zones         {counts['zones']:>5}          │")
        print(f"  │  Aisles        {counts['aisles']:>5}          │")
        print(f"  │  Slots         {counts['slots']:>5}          │")
        print(f"  │  Robots        {counts['robots']:>5}          │")
        print(f"  │  Regulations   {counts['regulations']:>5}          │")
        print(f"  │  Relationships {counts['relationships']:>5}          │")
        print("  └─────────────────────────────┘")


def main() -> None:
    print(f"\nConnecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected.\n")
        t0 = time.time()
        seed(driver)
        print(f"\nDone in {time.time() - t0:.1f}s\n")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
