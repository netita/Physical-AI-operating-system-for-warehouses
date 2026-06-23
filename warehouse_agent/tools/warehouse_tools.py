"""
Warehouse tool definitions and handlers for the WarehouseGPT agent.

All tools follow the OpenAI SDK function-calling format with JSON Schema parameters.
Handlers are async callables that receive the tool input dict and return a string result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DIGITAL_TWIN_URL = os.environ.get("DIGITAL_TWIN_URL", "http://localhost:8003")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real warehouse slot data
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).parent.parent.parent / "user_data" / "info_slots.json"

try:
    with open(_DATA_FILE) as _f:
        _SLOTS: Dict[str, List[str]] = json.load(_f)
    # Build reverse index: box_id -> slot_id
    _BOX_TO_SLOT: Dict[str, str] = {
        box: slot for slot, boxes in _SLOTS.items() for box in boxes
    }
    logger.info("Loaded %d slots, %d boxes from %s", len(_SLOTS), len(_BOX_TO_SLOT), _DATA_FILE)
except Exception as _e:
    logger.warning("Could not load slot data from %s: %s", _DATA_FILE, _e)
    _SLOTS = {}
    _BOX_TO_SLOT = {}

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_inventory_status",
            "description": (
                "Retrieve current inventory levels for items in the warehouse. "
                "Returns stock counts, locations, and reorder alerts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SKU/item IDs to check. Pass empty list for all items.",
                    },
                    "zone": {
                        "type": "string",
                        "description": "Optional warehouse zone filter (e.g. 'A', 'B', 'COLD-STORAGE').",
                    },
                    "include_reserved": {
                        "type": "boolean",
                        "description": "Whether to include reserved/allocated stock in counts.",
                    },
                },
                "required": ["item_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_robot_task",
            "description": (
                "Assign a task to an autonomous mobile robot (AMR) or forklift robot. "
                "Tasks include pick, place, transport, charge, and patrol operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "robot_id": {
                        "type": "string",
                        "description": "Unique robot identifier (e.g. 'AMR-001', 'FORK-003').",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["pick", "place", "transport", "charge", "patrol", "emergency_stop"],
                        "description": "The type of task to assign.",
                    },
                    "location_from": {
                        "type": "string",
                        "description": "Source location code (e.g. 'A-12-3', 'DOCK-2').",
                    },
                    "location_to": {
                        "type": "string",
                        "description": "Destination location code.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Task priority (1=lowest, 10=highest/emergency).",
                    },
                    "payload": {
                        "type": "object",
                        "description": "Optional task payload (item_id, quantity, notes).",
                    },
                },
                "required": ["robot_id", "task_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_fleet_status",
            "description": (
                "Get real-time status of the robot fleet including battery levels, "
                "current tasks, positions, and any fault conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "robot_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Robot IDs to query. Empty list returns all robots.",
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "description": "Include performance metrics (tasks completed, uptime).",
                    },
                },
                "required": ["robot_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_safety_regulations",
            "description": (
                "Query OSHA, ISO 3691, and internal safety regulations relevant to warehouse "
                "operations. Returns applicable rules, required PPE, clearance distances, "
                "speed limits, and compliance checklists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Safety regulation query (e.g. 'forklift pedestrian separation').",
                    },
                    "regulation_type": {
                        "type": "string",
                        "enum": ["OSHA", "ISO_3691", "INTERNAL", "ALL"],
                        "description": "Filter by regulation type.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Operational context for relevance filtering.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_safety_incident",
            "description": (
                "Report a safety incident, near-miss, or hazard observation. "
                "Triggers immediate notifications, logs to incident database, "
                "and may initiate emergency protocols."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_type": {
                        "type": "string",
                        "enum": [
                            "collision",
                            "near_miss",
                            "equipment_failure",
                            "hazard_observation",
                            "injury",
                            "fire",
                            "spill",
                        ],
                        "description": "Type of safety incident.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Incident severity level.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Warehouse location where incident occurred.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the incident.",
                    },
                    "robots_involved": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of any robots involved.",
                    },
                    "immediate_action_required": {
                        "type": "boolean",
                        "description": "Whether immediate emergency response is needed.",
                    },
                },
                "required": ["incident_type", "severity", "location", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_warehouse_layout",
            "description": (
                "Analyze and suggest optimizations for warehouse layout, robot traffic patterns, "
                "pick paths, and storage slot assignments based on order frequency and weight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "optimization_target": {
                        "type": "string",
                        "enum": [
                            "throughput",
                            "energy_efficiency",
                            "space_utilization",
                            "safety",
                            "balanced",
                        ],
                        "description": "Primary optimization objective.",
                    },
                    "zones_to_analyze": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific zones to optimize. Empty = full warehouse.",
                    },
                    "time_horizon_days": {
                        "type": "integer",
                        "description": "Historical data period in days for analysis.",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional constraints (max_robot_speed, fixed_locations, etc.).",
                    },
                },
                "required": ["optimization_target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Retrieve status of warehouse orders including picking progress, "
                "robot assignments, estimated completion times, and exception flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Order IDs to query. Empty = pending/active orders.",
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "exception", "all"],
                        "description": "Filter orders by status.",
                    },
                    "include_robot_assignments": {
                        "type": "boolean",
                        "description": "Include robot assignment details for each order.",
                    },
                },
                "required": ["order_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_item_location",
            "description": (
                "Find the exact warehouse slot location of a box or item by its ID. "
                "Returns the slot address in ZONE/AISLE/LEVEL/BAY/POSITION format "
                "and all other boxes stored in the same slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "box_id": {
                        "type": "string",
                        "description": "The box or item ID to locate (e.g. '1234').",
                    },
                },
                "required": ["box_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_slot_contents",
            "description": (
                "Get all boxes stored in a specific warehouse slot. "
                "Slot address format: ZONE/AISLE/LEVEL/BAY/POSITION (e.g. 'A/01/0/0/01'). "
                "Also supports listing all slots in a zone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {
                        "type": "string",
                        "description": "Slot address (e.g. 'A/03/1/2/02') or zone letter (e.g. 'A').",
                    },
                },
                "required": ["slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_positions",
            "description": (
                "Get real-time positions of all agents in the warehouse from the live digital twin. "
                "Returns current locations of forklifts, workers, and AMRs with coordinates, "
                "headings, and velocities. Use this when asked about where robots or workers are right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_type": {
                        "type": "string",
                        "enum": ["forklift", "worker", "amr", "all"],
                        "description": "Filter by agent type. Default is all.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_incidents",
            "description": (
                "Get active safety incidents currently happening in the warehouse from the live digital twin. "
                "Returns incident type, severity, location, and involved agents. "
                "Use this when asked about current alerts, safety issues, or ongoing incidents."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearest_agent",
            "description": (
                "Find which warehouse agent (forklift, worker, or AMR) is closest to a given "
                "position or warehouse slot. Useful for dispatching the nearest available robot "
                "or checking if any agent is near a specific location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X coordinate in metres (0–100).",
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate in metres (0–60).",
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["forklift", "worker", "amr", "all"],
                        "description": "Only consider agents of this type.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of nearest agents to return (default 3).",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_diagnostic",
            "description": (
                "Run system diagnostics on warehouse infrastructure including robot health checks, "
                "sensor calibration status, network connectivity, and WMS integration health."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": [
                            "robots",
                            "sensors",
                            "network",
                            "wms",
                            "charging_stations",
                            "safety_systems",
                            "all",
                        ],
                        "description": "System component to diagnose.",
                    },
                    "deep_scan": {
                        "type": "boolean",
                        "description": "Run extended diagnostics (takes longer).",
                    },
                    "robot_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific robot IDs for targeted robot diagnostics.",
                    },
                },
                "required": ["component"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Simulated handler implementations
# In production these would integrate with the WMS, fleet manager, and sensor layer.
# ---------------------------------------------------------------------------


async def _handle_get_inventory_status(tool_input: Dict[str, Any]) -> str:
    """Retrieve inventory levels from real slot data."""
    await asyncio.sleep(0.01)
    item_ids: List[str] = tool_input.get("item_ids", [])
    zone_filter: str = tool_input.get("zone", "ALL").upper()
    include_reserved: bool = tool_input.get("include_reserved", False)

    slots = _SLOTS
    if zone_filter != "ALL":
        slots = {k: v for k, v in slots.items() if k.startswith(zone_filter + "/")}

    if not item_ids:
        # Warehouse summary from real data
        zones: Dict[str, int] = {}
        total_boxes = 0
        for slot_id, boxes in slots.items():
            z = slot_id.split("/")[0]
            zones[z] = zones.get(z, 0) + len(boxes)
            total_boxes += len(boxes)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zone_filter": zone_filter,
            "data_source": "real",
            "summary": {
                "total_slots": len(slots),
                "total_boxes": total_boxes,
                "zones": zones,
                "avg_boxes_per_slot": round(total_boxes / len(slots), 1) if slots else 0,
            },
        }
    else:
        items = []
        for box_id in item_ids:
            slot = _BOX_TO_SLOT.get(str(box_id))
            if slot:
                slot_boxes = _SLOTS.get(slot, [])
                items.append({
                    "box_id": box_id,
                    "found": True,
                    "slot": slot,
                    "zone": slot.split("/")[0],
                    "aisle": slot.split("/")[1],
                    "level": slot.split("/")[2],
                    "bay": slot.split("/")[3],
                    "position": slot.split("/")[4],
                    "co_located_boxes": len(slot_boxes) - 1,
                    "reserved": include_reserved,
                })
            else:
                items.append({"box_id": box_id, "found": False, "slot": None})

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "real",
            "queried": len(item_ids),
            "found": sum(1 for i in items if i["found"]),
            "items": items,
        }

    return json.dumps(result, indent=2)


async def _handle_assign_robot_task(tool_input: Dict[str, Any]) -> str:
    """Assign task to a robot — simulates WMS task dispatch."""
    await asyncio.sleep(0.05)
    robot_id: str = tool_input["robot_id"]
    task_type: str = tool_input["task_type"]
    location_from: str = tool_input.get("location_from", "")
    location_to: str = tool_input.get("location_to", "")
    priority: int = tool_input.get("priority", 5)

    task_id = f"TASK-{random.randint(10000, 99999)}"

    if task_type == "emergency_stop":
        result = {
            "status": "EMERGENCY_STOP_INITIATED",
            "robot_id": robot_id,
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Emergency stop command sent to {robot_id}. Robot halted immediately.",
            "safety_alert": True,
        }
    else:
        eta_minutes = random.randint(2, 15)
        result = {
            "status": "TASK_ASSIGNED",
            "robot_id": robot_id,
            "task_id": task_id,
            "task_type": task_type,
            "location_from": location_from,
            "location_to": location_to,
            "priority": priority,
            "estimated_completion_minutes": eta_minutes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queue_position": random.randint(1, 3),
        }

    return json.dumps(result, indent=2)


async def _handle_get_robot_fleet_status(tool_input: Dict[str, Any]) -> str:
    """Return fleet status for requested robots."""
    await asyncio.sleep(0.05)
    robot_ids: List[str] = tool_input.get("robot_ids", [])
    include_metrics: bool = tool_input.get("include_metrics", False)

    fleet_data: List[Dict[str, Any]] = []
    all_robots = robot_ids if robot_ids else [
        "AMR-001", "AMR-002", "AMR-003", "AMR-004",
        "FORK-001", "FORK-002", "FORK-003",
    ]

    for rid in all_robots:
        battery = random.randint(15, 100)
        statuses = ["idle", "picking", "transporting", "charging", "returning_to_base"]
        status = "charging" if battery < 20 else random.choice(statuses)
        robot: Dict[str, Any] = {
            "robot_id": rid,
            "type": "AMR" if rid.startswith("AMR") else "Forklift",
            "status": status,
            "battery_pct": battery,
            "battery_warning": battery < 25,
            "position": {
                "zone": random.choice(["A", "B", "C"]),
                "x_m": round(random.uniform(0, 100), 1),
                "y_m": round(random.uniform(0, 50), 1),
            },
            "current_task": f"TASK-{random.randint(10000, 99999)}" if status not in ("idle", "charging") else None,
            "fault_codes": [],
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        }
        if include_metrics:
            robot["metrics"] = {
                "tasks_completed_today": random.randint(10, 80),
                "uptime_hours_today": round(random.uniform(4, 10), 1),
                "distance_km_today": round(random.uniform(5, 30), 1),
                "error_count_today": random.randint(0, 3),
            }
        fleet_data.append(robot)

    return json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fleet_size": len(fleet_data),
            "active_robots": sum(1 for r in fleet_data if r["status"] not in ("idle", "charging")),
            "robots": fleet_data,
        },
        indent=2,
    )


async def _handle_query_safety_regulations(tool_input: Dict[str, Any]) -> str:
    """Return safety regulations matching the query."""
    await asyncio.sleep(0.05)
    query: str = tool_input["query"]
    regulation_type: str = tool_input.get("regulation_type", "ALL")

    # Simulated regulation knowledge base lookup
    regulations = {
        "forklift": [
            {
                "id": "OSHA-1910.178(l)",
                "type": "OSHA",
                "title": "Powered Industrial Truck Operator Training",
                "summary": "Operators must be trained and certified before operating forklifts.",
                "key_requirements": [
                    "Formal instruction (lecture, discussion, written material)",
                    "Practical training (demonstrations, exercises)",
                    "Evaluation under actual conditions",
                    "Refresher training when unsafe operation observed",
                ],
            },
            {
                "id": "ISO-3691-4:2020",
                "type": "ISO_3691",
                "title": "Industrial Trucks — Safety Requirements",
                "summary": "Safety requirements and verification for driverless industrial trucks.",
                "key_requirements": [
                    "Minimum 0.5m safety clearance around robot path",
                    "Emergency stop devices within operator reach",
                    "Speed limited to 1.2 m/s in pedestrian zones",
                    "Visual and auditory warning signals required",
                ],
            },
        ],
        "pedestrian": [
            {
                "id": "OSHA-1910.176",
                "type": "OSHA",
                "title": "Material Handling — Pedestrian Safety",
                "summary": "Requires physical separation or right-of-way rules for pedestrians.",
                "key_requirements": [
                    "Designated pedestrian walkways with floor markings",
                    "4-way stop signs at intersections",
                    "Maximum robot speed 1.0 m/s near pedestrian zones",
                    "Proximity sensors mandatory for robot-pedestrian detection",
                ],
            },
        ],
        "fire": [
            {
                "id": "OSHA-1910.157",
                "type": "OSHA",
                "title": "Portable Fire Extinguishers",
                "summary": "Fire extinguisher placement, inspection, and training requirements.",
                "key_requirements": [
                    "Extinguisher within 75 feet travel distance",
                    "Monthly visual inspections",
                    "Annual maintenance checks",
                    "Annual employee training",
                ],
            },
        ],
    }

    # Simple keyword matching
    matched: List[Dict[str, Any]] = []
    query_lower = query.lower()
    for keyword, regs in regulations.items():
        if keyword in query_lower:
            for reg in regs:
                if regulation_type == "ALL" or reg["type"] == regulation_type:
                    matched.append(reg)

    if not matched:
        matched = [
            {
                "id": "GENERAL-001",
                "type": "INTERNAL",
                "title": "General Warehouse Safety Policy",
                "summary": "All personnel must follow posted safety signs and robot right-of-way rules.",
                "key_requirements": [
                    "Wear high-visibility vest in active robot zones",
                    "Never obstruct robot pathways",
                    "Report all near-misses within 15 minutes",
                    "Mandatory safety briefing for new personnel",
                ],
            }
        ]

    return json.dumps(
        {
            "query": query,
            "regulation_type_filter": regulation_type,
            "regulations_found": len(matched),
            "regulations": matched,
            "disclaimer": "Always verify with your HSE officer. Regulations may have been updated.",
        },
        indent=2,
    )


async def _handle_report_safety_incident(tool_input: Dict[str, Any]) -> str:
    """Log and escalate a safety incident."""
    await asyncio.sleep(0.1)
    incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    severity: str = tool_input["severity"]
    immediate: bool = tool_input.get("immediate_action_required", False)
    robots_involved: List[str] = tool_input.get("robots_involved", [])

    actions_taken: List[str] = ["Incident logged to safety database", "HSE officer notified via email"]

    if severity in ("high", "critical"):
        actions_taken.extend([
            "Site safety manager alerted via SMS",
            "OSHA incident log updated",
        ])
        if robots_involved:
            actions_taken.append(
                f"Robots {', '.join(robots_involved)} suspended pending investigation"
            )

    if immediate or severity == "critical":
        actions_taken.extend([
            "EMERGENCY ALERT broadcast to all floor personnel",
            "Zone cordoned off automatically",
            "Emergency response team dispatched",
        ])

    return json.dumps(
        {
            "incident_id": incident_id,
            "status": "REPORTED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "actions_taken": actions_taken,
            "follow_up_required": severity in ("high", "critical"),
            "investigation_deadline": (
                "24 hours" if severity == "critical" else "72 hours" if severity == "high" else "7 days"
            ),
            "message": (
                f"Incident {incident_id} has been reported and escalated appropriately."
            ),
        },
        indent=2,
    )


async def _handle_optimize_warehouse_layout(tool_input: Dict[str, Any]) -> str:
    """Generate layout optimization recommendations."""
    await asyncio.sleep(0.1)
    target: str = tool_input["optimization_target"]
    zones: List[str] = tool_input.get("zones_to_analyze", ["ALL"])
    days: int = tool_input.get("time_horizon_days", 30)

    recommendations: Dict[str, Any] = {
        "throughput": {
            "headline": "Increase throughput by ~18% through path optimization",
            "actions": [
                "Relocate top-50 fastest-moving SKUs to Zone A pick faces",
                "Implement bidirectional traffic corridors in Zone B",
                "Add 2 additional AMR charging stations near Zone C",
                "Create express pick lanes for single-item orders",
            ],
            "estimated_improvement": "18.2% throughput increase",
            "implementation_effort": "Medium (2-3 days downtime)",
        },
        "energy_efficiency": {
            "headline": "Reduce robot energy consumption by ~22%",
            "actions": [
                "Cluster charging stations at geometric center of travel paths",
                "Implement zone-based task batching to reduce empty travel",
                "Schedule deep-clean operations during low-traffic periods",
                "Reduce AMR max speed to 1.0 m/s during off-peak hours",
            ],
            "estimated_improvement": "22.4% energy reduction",
            "implementation_effort": "Low (software configuration only)",
        },
        "space_utilization": {
            "headline": "Improve storage density by ~15%",
            "actions": [
                "Convert 12 static pick faces to dynamic slotting in Zone B",
                "Stack slow-movers to upper tiers (robot retrieval only)",
                "Consolidate partial pallets in Zone C",
                "Implement ABC velocity slotting across all zones",
            ],
            "estimated_improvement": "15.1% space utilization increase",
            "implementation_effort": "High (3-5 days reslotting)",
        },
        "safety": {
            "headline": "Eliminate 3 high-risk pedestrian-robot conflict zones",
            "actions": [
                "Install physical barriers at Zone A/B intersection",
                "Widen pedestrian corridor in Zone C from 1.5m to 2.2m",
                "Add redundant proximity sensors at dock doors 3, 5, 7",
                "Implement one-way robot traffic flow in Zone B aisle 4",
            ],
            "estimated_improvement": "Projected 71% reduction in near-miss incidents",
            "implementation_effort": "Medium (1-2 days, requires safety shutdown)",
        },
        "balanced": {
            "headline": "Balanced optimization across all dimensions",
            "actions": [
                "Velocity-based slotting for top-200 SKUs",
                "Robot path consolidation reducing average travel distance by 12%",
                "Two additional safety barriers at high-traffic intersections",
                "Charging station relocation to reduce deadhead travel",
            ],
            "estimated_improvement": "12% throughput, 8% energy, 10% safety score",
            "implementation_effort": "Medium (2-3 days phased rollout)",
        },
    }

    rec = recommendations.get(target, recommendations["balanced"])

    return json.dumps(
        {
            "optimization_target": target,
            "zones_analyzed": zones,
            "data_period_days": days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendation": rec,
            "roi_payback_months": random.randint(3, 12),
            "confidence_score": round(random.uniform(0.78, 0.95), 2),
        },
        indent=2,
    )


async def _handle_get_order_status(tool_input: Dict[str, Any]) -> str:
    """Return order fulfillment status."""
    await asyncio.sleep(0.05)
    order_ids: List[str] = tool_input.get("order_ids", [])
    status_filter: str = tool_input.get("status_filter", "all")
    include_robots: bool = tool_input.get("include_robot_assignments", True)

    statuses = ["pending", "in_progress", "completed", "exception"]
    if not order_ids:
        order_ids = [f"ORD-{random.randint(10000, 99999)}" for _ in range(5)]

    orders: List[Dict[str, Any]] = []
    for oid in order_ids:
        status = random.choice(statuses)
        if status_filter != "all" and status != status_filter:
            status = status_filter
        items_total = random.randint(1, 20)
        items_picked = (
            items_total if status == "completed"
            else 0 if status == "pending"
            else random.randint(0, items_total)
        )
        order: Dict[str, Any] = {
            "order_id": oid,
            "status": status,
            "items_total": items_total,
            "items_picked": items_picked,
            "progress_pct": round(items_picked / items_total * 100),
            "priority": random.choice(["standard", "express", "same-day"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "estimated_completion": datetime.now(timezone.utc).isoformat(),
            "exception_reason": (
                "Item SKU-4821 out of stock" if status == "exception" else None
            ),
        }
        if include_robots and status == "in_progress":
            order["assigned_robots"] = [
                {"robot_id": f"AMR-00{random.randint(1,4)}", "task": "picking"}
            ]
        orders.append(order)

    return json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "orders_count": len(orders),
            "status_filter": status_filter,
            "orders": orders,
        },
        indent=2,
    )


async def _handle_run_diagnostic(tool_input: Dict[str, Any]) -> str:
    """Run system diagnostic checks."""
    await asyncio.sleep(0.2 if tool_input.get("deep_scan") else 0.05)
    component: str = tool_input["component"]
    deep_scan: bool = tool_input.get("deep_scan", False)
    robot_ids: List[str] = tool_input.get("robot_ids", [])

    diagnostics: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "deep_scan": deep_scan,
        "overall_health": "healthy",
        "checks": [],
    }

    component_checks: Dict[str, List[Dict[str, Any]]] = {
        "robots": [
            {"name": "Firmware version", "status": "pass", "details": "All robots on v4.2.1"},
            {"name": "Motor calibration", "status": "pass", "details": "All within tolerance"},
            {"name": "Battery health", "status": "warning", "details": "AMR-003 battery at 87% capacity (replace in 30 days)"},
            {"name": "Lidar sensors", "status": "pass", "details": "All lidar arrays nominal"},
            {"name": "Emergency stop", "status": "pass", "details": "All e-stop circuits verified"},
        ],
        "sensors": [
            {"name": "Proximity sensors", "status": "pass", "details": "48/48 sensors online"},
            {"name": "Temperature sensors", "status": "warning", "details": "Zone B-4 sensor reading +2°C drift"},
            {"name": "Load cells", "status": "pass", "details": "All calibrated within 0.5% tolerance"},
            {"name": "Barcode scanners", "status": "pass", "details": "12/12 scanners functional"},
        ],
        "network": [
            {"name": "WiFi coverage", "status": "pass", "details": "95% warehouse coverage, avg -67 dBm"},
            {"name": "WMS connection", "status": "pass", "details": "Latency 12ms"},
            {"name": "Robot heartbeat", "status": "pass", "details": "All robots responding < 100ms"},
            {"name": "Cloud sync", "status": "pass", "details": "Last sync 3 minutes ago"},
        ],
        "wms": [
            {"name": "Database connectivity", "status": "pass", "details": "PostgreSQL responding in 5ms"},
            {"name": "Inventory sync", "status": "pass", "details": "In sync"},
            {"name": "Order queue", "status": "pass", "details": "23 orders in queue"},
            {"name": "API health", "status": "pass", "details": "All endpoints responding"},
        ],
        "charging_stations": [
            {"name": "Station CS-01", "status": "pass", "details": "Operational, 0 robots charging"},
            {"name": "Station CS-02", "status": "pass", "details": "Operational, 2 robots charging"},
            {"name": "Station CS-03", "status": "fail", "details": "Connection fault — maintenance required"},
            {"name": "Station CS-04", "status": "pass", "details": "Operational, 1 robot charging"},
        ],
        "safety_systems": [
            {"name": "E-stop network", "status": "pass", "details": "All 24 e-stop buttons verified"},
            {"name": "Fire suppression", "status": "pass", "details": "System armed and ready"},
            {"name": "Sprinkler system", "status": "pass", "details": "Pressure nominal"},
            {"name": "Safety light curtains", "status": "pass", "details": "All 8 curtains active"},
        ],
    }

    if component == "all":
        all_checks: List[Dict[str, Any]] = []
        for comp_checks in component_checks.values():
            all_checks.extend(comp_checks)
        diagnostics["checks"] = all_checks
    else:
        diagnostics["checks"] = component_checks.get(component, [])

    # Determine overall health
    statuses_found = [c["status"] for c in diagnostics["checks"]]
    if "fail" in statuses_found:
        diagnostics["overall_health"] = "degraded"
    elif "warning" in statuses_found:
        diagnostics["overall_health"] = "warning"

    if robot_ids:
        diagnostics["targeted_robots"] = robot_ids

    diagnostics["issues_found"] = sum(1 for c in diagnostics["checks"] if c["status"] != "pass")
    diagnostics["recommendations"] = (
        ["Schedule maintenance for CS-03 charging station", "Monitor AMR-003 battery health"]
        if diagnostics["overall_health"] != "healthy"
        else ["System operating within normal parameters"]
    )

    return json.dumps(diagnostics, indent=2)


async def _handle_find_item_location(tool_input: Dict[str, Any]) -> str:
    """Find exact slot location of a box by ID."""
    await asyncio.sleep(0.01)
    box_id = str(tool_input.get("box_id", "")).strip()

    slot = _BOX_TO_SLOT.get(box_id)
    if not slot:
        return json.dumps({
            "box_id": box_id,
            "found": False,
            "message": f"Box '{box_id}' not found in any warehouse slot.",
        }, indent=2)

    parts = slot.split("/")
    slot_boxes = _SLOTS.get(slot, [])

    return json.dumps({
        "box_id": box_id,
        "found": True,
        "slot": slot,
        "zone": parts[0],
        "aisle": parts[1],
        "level": parts[2],
        "bay": parts[3],
        "position": parts[4],
        "all_boxes_in_slot": slot_boxes,
        "total_boxes_in_slot": len(slot_boxes),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


async def _handle_get_slot_contents(tool_input: Dict[str, Any]) -> str:
    """Return contents of a specific slot or summary of a zone."""
    await asyncio.sleep(0.01)
    slot_id = str(tool_input.get("slot_id", "")).strip().upper()

    # Single slot lookup
    if slot_id in _SLOTS:
        boxes = _SLOTS[slot_id]
        return json.dumps({
            "slot": slot_id,
            "boxes": boxes,
            "box_count": len(boxes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2)

    # Zone summary (single letter like "A" or "B")
    if len(slot_id) == 1:
        zone_slots = {k: v for k, v in _SLOTS.items() if k.startswith(slot_id + "/")}
        if zone_slots:
            total = sum(len(v) for v in zone_slots.values())
            occupied = sum(1 for v in zone_slots.values() if v)
            empty = len(zone_slots) - occupied

            # Sample of slots with most boxes
            top_slots = sorted(zone_slots.items(), key=lambda x: len(x[1]), reverse=True)[:5]

            return json.dumps({
                "zone": slot_id,
                "total_slots": len(zone_slots),
                "occupied_slots": occupied,
                "empty_slots": empty,
                "total_boxes": total,
                "avg_boxes_per_slot": round(total / len(zone_slots), 1) if zone_slots else 0,
                "fullest_slots": [{"slot": s, "box_count": len(b)} for s, b in top_slots],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, indent=2)

    return json.dumps({
        "slot_id": slot_id,
        "found": False,
        "message": f"Slot or zone '{slot_id}' not found. Use format 'A/01/0/0/01' or a zone letter like 'A'.",
        "available_zones": sorted({k.split("/")[0] for k in _SLOTS}),
    }, indent=2)


def _fetch_twin(path: str) -> Dict[str, Any]:
    """Synchronous HTTP GET from the digital twin API."""
    try:
        req = urllib.request.Request(f"{DIGITAL_TWIN_URL}{path}")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        return {"error": f"Digital twin unreachable: {exc.reason}"}
    except Exception as exc:
        return {"error": str(exc)}


async def _handle_get_live_positions(tool_input: Dict[str, Any]) -> str:
    """Fetch live agent positions from the digital twin."""
    agent_type: str = tool_input.get("agent_type", "all")

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_twin, "/state")

    if "error" in data:
        return json.dumps({
            "error": data["error"],
            "hint": "Make sure the digital twin API is running on port 8003.",
        }, indent=2)

    result: Dict[str, Any] = {
        "timestamp": data.get("timestamp"),
        "frame_index": data.get("frame_index"),
        "track_count": data.get("track_count", 0),
        "source": "live_digital_twin",
    }

    if agent_type in ("forklift", "all"):
        result["forklifts"] = data.get("forklifts", [])
    if agent_type in ("worker", "all"):
        result["workers"] = data.get("workers", [])
    if agent_type in ("amr", "all"):
        result["amrs"] = data.get("amrs", [])

    counts = {
        "forklifts": len(data.get("forklifts", [])),
        "workers": len(data.get("workers", [])),
        "amrs": len(data.get("amrs", [])),
    }
    result["summary"] = counts

    return json.dumps(result, indent=2)


async def _handle_get_live_incidents(tool_input: Dict[str, Any]) -> str:
    """Fetch active incidents from the digital twin."""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_twin, "/incidents")

    if "error" in data:
        return json.dumps({
            "error": data["error"],
            "hint": "Make sure the digital twin API is running on port 8003.",
        }, indent=2)

    incidents = data.get("incidents", [])
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "live_digital_twin",
        "active_incident_count": len(incidents),
        "incidents": incidents,
    }

    if not incidents:
        result["status"] = "No active incidents — warehouse operating normally."
    else:
        critical = [i for i in incidents if i.get("severity") == "critical"]
        high = [i for i in incidents if i.get("severity") == "high"]
        if critical:
            result["alert"] = f"[SAFETY ALERT] {len(critical)} CRITICAL incident(s) active!"
        elif high:
            result["alert"] = f"{len(high)} HIGH severity incident(s) require attention."

    return json.dumps(result, indent=2)


async def _handle_find_nearest_agent(tool_input: Dict[str, Any]) -> str:
    """Find agents nearest to given coordinates."""
    x: float = float(tool_input["x"])
    y: float = float(tool_input["y"])
    agent_type: str = tool_input.get("agent_type", "all")
    max_results: int = int(tool_input.get("max_results", 3))

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_twin, "/state")

    if "error" in data:
        return json.dumps({"error": data["error"]}, indent=2)

    candidates: List[Dict[str, Any]] = []
    if agent_type in ("forklift", "all"):
        candidates += [{"type": "forklift", **a} for a in data.get("forklifts", [])]
    if agent_type in ("worker", "all"):
        candidates += [{"type": "worker", **a} for a in data.get("workers", [])]
    if agent_type in ("amr", "all"):
        candidates += [{"type": "amr", **a} for a in data.get("amrs", [])]

    if not candidates:
        return json.dumps({
            "query": {"x": x, "y": y, "agent_type": agent_type},
            "result": "No agents found in the digital twin.",
            "source": "live_digital_twin",
        }, indent=2)

    for agent in candidates:
        dx = agent.get("x", 0) - x
        dy = agent.get("y", 0) - y
        agent["distance_m"] = round(math.sqrt(dx * dx + dy * dy), 2)

    candidates.sort(key=lambda a: a["distance_m"])
    nearest = candidates[:max_results]

    return json.dumps({
        "query": {"x": x, "y": y, "agent_type": agent_type},
        "source": "live_digital_twin",
        "nearest_agents": nearest,
        "closest": nearest[0] if nearest else None,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool handler dispatch table
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "get_inventory_status": _handle_get_inventory_status,
    "assign_robot_task": _handle_assign_robot_task,
    "get_robot_fleet_status": _handle_get_robot_fleet_status,
    "query_safety_regulations": _handle_query_safety_regulations,
    "report_safety_incident": _handle_report_safety_incident,
    "optimize_warehouse_layout": _handle_optimize_warehouse_layout,
    "get_order_status": _handle_get_order_status,
    "run_diagnostic": _handle_run_diagnostic,
    "find_item_location": _handle_find_item_location,
    "get_slot_contents": _handle_get_slot_contents,
    "get_live_positions": _handle_get_live_positions,
    "get_live_incidents": _handle_get_live_incidents,
    "find_nearest_agent": _handle_find_nearest_agent,
}
