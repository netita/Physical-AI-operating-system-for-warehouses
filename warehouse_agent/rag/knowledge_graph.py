"""
Warehouse Knowledge Graph using Neo4j.

Models typed nodes and relationships for the warehouse domain:
- Robots, Zones, Items, Orders, Incidents, Regulations
- Causal chain queries: incident → root_cause → contributing_factors
- Semantic relationship traversal for agent reasoning
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Neo4j import
# ---------------------------------------------------------------------------

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import ServiceUnavailable

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not available; WarehouseKnowledgeGraph in stub mode.")

# ---------------------------------------------------------------------------
# Node and relationship type constants
# ---------------------------------------------------------------------------

NODE_ROBOT = "Robot"
NODE_ZONE = "Zone"
NODE_ITEM = "Item"
NODE_ORDER = "Order"
NODE_INCIDENT = "Incident"
NODE_REGULATION = "Regulation"
NODE_EQUIPMENT = "Equipment"
NODE_TASK = "Task"

REL_ASSIGNED_TO = "ASSIGNED_TO"
REL_LOCATED_IN = "LOCATED_IN"
REL_CAUSED_BY = "CAUSED_BY"
REL_VIOLATED = "VIOLATED"
REL_INVOLVES = "INVOLVES"
REL_CONTAINS = "CONTAINS"
REL_PART_OF = "PART_OF"
REL_REQUIRED_BY = "REQUIRED_BY"
REL_PERFORMED = "PERFORMED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class GraphNode(object):
    """A node retrieved from the knowledge graph."""

    def __init__(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> None:
        self.node_id = node_id
        self.labels = labels
        self.properties = properties

    def __repr__(self) -> str:
        return f"GraphNode(labels={self.labels}, id={self.node_id!r})"


class GraphRelationship(object):
    """A relationship retrieved from the knowledge graph."""

    def __init__(
        self,
        rel_type: str,
        start_node: GraphNode,
        end_node: GraphNode,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rel_type = rel_type
        self.start_node = start_node
        self.end_node = end_node
        self.properties = properties or {}


class CausalChain(object):
    """A causal chain from incident to root causes."""

    def __init__(
        self,
        incident_id: str,
        incident_type: str,
        severity: str,
        causal_chain: List[Dict[str, Any]],
        contributing_factors: List[str],
        regulations_violated: List[str],
        recommended_actions: List[str],
    ) -> None:
        self.incident_id = incident_id
        self.incident_type = incident_type
        self.severity = severity
        self.causal_chain = causal_chain
        self.contributing_factors = contributing_factors
        self.regulations_violated = regulations_violated
        self.recommended_actions = recommended_actions

    def to_summary(self) -> str:
        """Format causal chain as a readable summary."""
        lines = [
            f"Incident {self.incident_id} ({self.incident_type}, {self.severity.upper()})",
            "",
            "Causal Chain:",
        ]
        for i, step in enumerate(self.causal_chain, 1):
            lines.append(f"  {i}. {step.get('description', str(step))}")

        if self.contributing_factors:
            lines.append("\nContributing Factors:")
            for factor in self.contributing_factors:
                lines.append(f"  - {factor}")

        if self.regulations_violated:
            lines.append("\nRegulations Violated:")
            for reg in self.regulations_violated:
                lines.append(f"  - {reg}")

        if self.recommended_actions:
            lines.append("\nRecommended Actions:")
            for action in self.recommended_actions:
                lines.append(f"  - {action}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WarehouseKnowledgeGraph implementation
# ---------------------------------------------------------------------------


class WarehouseKnowledgeGraph:
    """
    Knowledge graph for warehouse operations using Neo4j.

    Provides:
    - Node creation and relationship management
    - Causal chain analysis for safety incidents
    - Robot-task-zone relationship queries
    - Regulation compliance checking
    - Falls back to in-memory stub when Neo4j is unavailable
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "warehouse_ai",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        self._driver: Optional[Any] = None
        self._available = False
        self._stub_nodes: Dict[str, Dict[str, Any]] = {}
        self._stub_rels: List[Dict[str, Any]] = []

        self._init_driver()
        self._seed_stub_data()

    def _init_driver(self) -> None:
        """Initialize Neo4j async driver."""
        if not NEO4J_AVAILABLE:
            logger.info("WarehouseKnowledgeGraph: operating in stub mode (no neo4j driver)")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
            )
            self._available = True
            logger.info("WarehouseKnowledgeGraph: Neo4j driver initialized at %s", self.uri)
        except Exception as exc:
            logger.warning(
                "Neo4j connection failed (%s), using stub mode: %s",
                self.uri,
                exc,
            )
            self._available = False

    def _seed_stub_data(self) -> None:
        """Pre-populate stub data for development without Neo4j."""
        self._stub_nodes = {
            "AMR-001": {"type": NODE_ROBOT, "id": "AMR-001", "zone": "A", "status": "active", "battery": 85},
            "AMR-002": {"type": NODE_ROBOT, "id": "AMR-002", "zone": "B", "status": "active", "battery": 62},
            "AMR-003": {"type": NODE_ROBOT, "id": "AMR-003", "zone": "C", "status": "charging", "battery": 22},
            "FORK-001": {"type": NODE_ROBOT, "id": "FORK-001", "zone": "B", "status": "picking", "battery": 75},
            "ZONE-A": {"type": NODE_ZONE, "id": "ZONE-A", "name": "Receiving & Fast-Pick", "area_m2": 2500},
            "ZONE-B": {"type": NODE_ZONE, "id": "ZONE-B", "name": "Bulk Storage", "area_m2": 5000},
            "ZONE-C": {"type": NODE_ZONE, "id": "ZONE-C", "name": "Cold Storage", "area_m2": 1500},
            "ZONE-D": {"type": NODE_ZONE, "id": "ZONE-D", "name": "Outbound Staging", "area_m2": 2000},
            "INC-001": {
                "type": NODE_INCIDENT,
                "id": "INC-001",
                "incident_type": "near_miss",
                "severity": "medium",
                "location": "Zone B, Aisle 4",
                "description": "AMR-002 approached pedestrian crossing faster than allowed speed limit",
                "timestamp": "2026-06-15T14:32:00Z",
                "resolved": False,
            },
            "REG-OSHA-178": {
                "type": NODE_REGULATION,
                "id": "REG-OSHA-178",
                "source": "OSHA",
                "code": "1910.178(l)",
                "title": "Powered Industrial Truck Operator Training",
            },
            "REG-ISO-3691": {
                "type": NODE_REGULATION,
                "id": "REG-ISO-3691",
                "source": "ISO",
                "code": "3691-4:2020",
                "title": "Driverless Industrial Trucks Safety Requirements",
            },
        }
        self._stub_rels = [
            {"type": REL_LOCATED_IN, "from": "AMR-001", "to": "ZONE-A"},
            {"type": REL_LOCATED_IN, "from": "AMR-002", "to": "ZONE-B"},
            {"type": REL_LOCATED_IN, "from": "AMR-003", "to": "ZONE-C"},
            {"type": REL_LOCATED_IN, "from": "FORK-001", "to": "ZONE-B"},
            {"type": REL_INVOLVES, "from": "INC-001", "to": "AMR-002"},
            {"type": REL_CAUSED_BY, "from": "INC-001", "to": "REG-ISO-3691"},
            {"type": REL_VIOLATED, "from": "INC-001", "to": "REG-ISO-3691"},
        ]

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            await self._driver.close()
            logger.info("Neo4j driver closed.")

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    async def create_node(
        self,
        node_type: str,
        node_id: str,
        properties: Dict[str, Any],
    ) -> bool:
        """Create or update a node in the knowledge graph."""
        properties["id"] = node_id
        properties["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    query = (
                        f"MERGE (n:{node_type} {{id: $node_id}}) "
                        "SET n += $properties "
                        "RETURN n.id AS id"
                    )
                    result = await session.run(query, node_id=node_id, properties=properties)
                    record = await result.single()
                    return record is not None
            except Exception as exc:
                logger.error("Neo4j create_node failed: %s", exc)

        # Fallback
        self._stub_nodes[node_id] = {"type": node_type, **properties}
        return True

    async def create_relationship(
        self,
        from_id: str,
        rel_type: str,
        to_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two nodes."""
        props = properties or {}
        props["created_at"] = datetime.now(timezone.utc).isoformat()

        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    query = (
                        "MATCH (a {id: $from_id}), (b {id: $to_id}) "
                        f"MERGE (a)-[r:{rel_type}]->(b) "
                        "SET r += $properties "
                        "RETURN type(r) AS rel_type"
                    )
                    result = await session.run(
                        query, from_id=from_id, to_id=to_id, properties=props
                    )
                    record = await result.single()
                    return record is not None
            except Exception as exc:
                logger.error("Neo4j create_relationship failed: %s", exc)

        self._stub_rels.append({"type": rel_type, "from": from_id, "to": to_id, **props})
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_robot_context(self, robot_id: str) -> Dict[str, Any]:
        """
        Get full context for a robot: current zone, active tasks, recent incidents.
        """
        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    query = """
                    MATCH (r:Robot {id: $robot_id})
                    OPTIONAL MATCH (r)-[:LOCATED_IN]->(z:Zone)
                    OPTIONAL MATCH (t:Task)-[:ASSIGNED_TO]->(r)
                    OPTIONAL MATCH (i:Incident)-[:INVOLVES]->(r) WHERE i.resolved = false
                    RETURN
                        r AS robot,
                        z AS zone,
                        collect(DISTINCT t) AS tasks,
                        collect(DISTINCT i) AS incidents
                    """
                    result = await session.run(query, robot_id=robot_id)
                    record = await result.single()
                    if record:
                        return {
                            "robot": dict(record["robot"]) if record["robot"] else {},
                            "zone": dict(record["zone"]) if record["zone"] else {},
                            "active_tasks": [dict(t) for t in record["tasks"]],
                            "active_incidents": [dict(i) for i in record["incidents"]],
                        }
            except Exception as exc:
                logger.error("Neo4j get_robot_context failed: %s", exc)

        # Fallback
        robot_data = self._stub_nodes.get(robot_id, {})
        zone_id = None
        for rel in self._stub_rels:
            if rel["from"] == robot_id and rel["type"] == REL_LOCATED_IN:
                zone_id = rel["to"]
                break
        zone_data = self._stub_nodes.get(zone_id, {}) if zone_id else {}
        incidents = [
            self._stub_nodes[r["from"]]
            for r in self._stub_rels
            if r["type"] == REL_INVOLVES and r["to"] == robot_id
            and r["from"] in self._stub_nodes
            and not self._stub_nodes[r["from"]].get("resolved", True)
        ]

        return {
            "robot": robot_data,
            "zone": zone_data,
            "active_tasks": [],
            "active_incidents": incidents,
        }

    async def get_causal_chain(self, incident_id: str) -> Optional[CausalChain]:
        """
        Retrieve the causal chain for a safety incident.

        Traverses: Incident → CAUSED_BY → root causes → contributing factors
        Also fetches regulations violated and generates recommended actions.
        """
        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    query = """
                    MATCH (i:Incident {id: $incident_id})
                    OPTIONAL MATCH (i)-[:INVOLVES]->(r:Robot)
                    OPTIONAL MATCH (i)-[:VIOLATED]->(reg:Regulation)
                    OPTIONAL MATCH (i)-[:CAUSED_BY]->(cause)
                    RETURN
                        i AS incident,
                        collect(DISTINCT r) AS robots,
                        collect(DISTINCT reg) AS regulations,
                        collect(DISTINCT cause) AS causes
                    """
                    result = await session.run(query, incident_id=incident_id)
                    record = await result.single()
                    if record and record["incident"]:
                        inc = dict(record["incident"])
                        robots = [dict(r) for r in record["robots"]]
                        regs = [dict(r) for r in record["regulations"]]
                        causes = [dict(c) for c in record["causes"]]

                        return CausalChain(
                            incident_id=incident_id,
                            incident_type=inc.get("incident_type", "unknown"),
                            severity=inc.get("severity", "unknown"),
                            causal_chain=[
                                {"description": c.get("description", str(c))} for c in causes
                            ],
                            contributing_factors=[
                                f"Robot {r.get('id')} involved" for r in robots
                            ],
                            regulations_violated=[
                                f"{r.get('source')} {r.get('code')} — {r.get('title')}"
                                for r in regs
                            ],
                            recommended_actions=self._generate_recommendations(inc, regs),
                        )
            except Exception as exc:
                logger.error("Neo4j causal chain query failed: %s", exc)

        # Fallback: stub data
        incident = self._stub_nodes.get(incident_id)
        if not incident:
            return None

        involved_robots = [
            self._stub_nodes[r["to"]]
            for r in self._stub_rels
            if r["from"] == incident_id and r["type"] == REL_INVOLVES and r["to"] in self._stub_nodes
        ]
        violated_regs = [
            self._stub_nodes[r["to"]]
            for r in self._stub_rels
            if r["from"] == incident_id and r["type"] == REL_VIOLATED and r["to"] in self._stub_nodes
        ]

        return CausalChain(
            incident_id=incident_id,
            incident_type=incident.get("incident_type", "unknown"),
            severity=incident.get("severity", "unknown"),
            causal_chain=[
                {
                    "description": (
                        f"Robot operating above speed limit in pedestrian zone "
                        f"at {incident.get('location', 'unknown location')}"
                    )
                },
                {
                    "description": "Speed limit configuration not updated after zone reclassification"
                },
                {
                    "description": "Pedestrian zone sensor degradation not detected during last maintenance"
                },
            ],
            contributing_factors=[
                f"Robot {r.get('id', 'unknown')} battery low affecting sensor response time"
                for r in involved_robots
            ] + [
                "Shift change handover did not communicate zone speed limit change",
                "Speed monitoring alert threshold set too high",
            ],
            regulations_violated=[
                f"{r.get('source', '')} {r.get('code', '')} — {r.get('title', '')}"
                for r in violated_regs
            ],
            recommended_actions=self._generate_recommendations(incident, violated_regs),
        )

    def _generate_recommendations(
        self, incident: Dict[str, Any], regulations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate corrective action recommendations based on incident data."""
        recommendations: List[str] = []
        incident_type = incident.get("incident_type", "")
        severity = incident.get("severity", "low")

        if incident_type in ("near_miss", "collision"):
            recommendations.extend([
                "Immediately audit robot speed configurations in all pedestrian zones",
                "Verify zone classification maps match physical signage",
                "Test proximity sensor response times on all affected robots",
            ])

        if incident_type == "equipment_failure":
            recommendations.extend([
                "Remove affected robot from service pending full diagnostic",
                "Accelerate scheduled maintenance for other robots of same model",
                "Review maintenance log for skipped or overdue inspections",
            ])

        if severity in ("high", "critical"):
            recommendations.extend([
                "Conduct emergency all-hands safety briefing within 24 hours",
                "Commission independent safety audit of affected zones",
            ])

        if regulations:
            recommendations.append(
                "Review compliance with all cited regulations: "
                + ", ".join(r.get("code", "") for r in regulations)
            )

        if not recommendations:
            recommendations.append("Document incident and schedule review at next safety meeting")

        return recommendations

    async def find_related_incidents(
        self,
        location: Optional[str] = None,
        robot_id: Optional[str] = None,
        incident_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find incidents related by location, robot, or type."""
        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    conditions: List[str] = []
                    params: Dict[str, Any] = {"limit": limit}

                    if location:
                        conditions.append("i.location CONTAINS $location")
                        params["location"] = location
                    if incident_type:
                        conditions.append("i.incident_type = $incident_type")
                        params["incident_type"] = incident_type

                    where_clause = ""
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)

                    robot_match = ""
                    if robot_id:
                        robot_match = "MATCH (i)-[:INVOLVES]->(:Robot {id: $robot_id})"
                        params["robot_id"] = robot_id

                    query = f"""
                    MATCH (i:Incident)
                    {robot_match}
                    {where_clause}
                    RETURN i
                    ORDER BY i.timestamp DESC
                    LIMIT $limit
                    """
                    result = await session.run(query, **params)
                    records = await result.data()
                    return [dict(r["i"]) for r in records]
            except Exception as exc:
                logger.error("Neo4j find_related_incidents failed: %s", exc)

        # Fallback
        incidents = [
            node for node in self._stub_nodes.values()
            if node.get("type") == NODE_INCIDENT
        ]
        if location:
            incidents = [i for i in incidents if location.lower() in i.get("location", "").lower()]
        if incident_type:
            incidents = [i for i in incidents if i.get("incident_type") == incident_type]
        if robot_id:
            related_incident_ids = {
                r["from"] for r in self._stub_rels
                if r["type"] == REL_INVOLVES and r["to"] == robot_id
            }
            incidents = [i for i in incidents if i.get("id") in related_incident_ids]

        return incidents[:limit]

    async def get_zone_compliance_status(self, zone_id: str) -> Dict[str, Any]:
        """Get regulation compliance status for a warehouse zone."""
        if self._available and self._driver is not None:
            try:
                async with self._driver.session(database=self.database) as session:
                    query = """
                    MATCH (z:Zone {id: $zone_id})
                    OPTIONAL MATCH (i:Incident)-[:LOCATED_IN]->(z) WHERE i.resolved = false
                    OPTIONAL MATCH (reg:Regulation)-[:REQUIRED_BY]->(z)
                    RETURN
                        z AS zone,
                        count(DISTINCT i) AS open_incidents,
                        collect(DISTINCT reg) AS required_regulations
                    """
                    result = await session.run(query, zone_id=zone_id)
                    record = await result.single()
                    if record:
                        return {
                            "zone": dict(record["zone"]) if record["zone"] else {},
                            "open_incidents": record["open_incidents"],
                            "required_regulations": [
                                dict(r) for r in record["required_regulations"]
                            ],
                        }
            except Exception as exc:
                logger.error("Neo4j zone compliance query failed: %s", exc)

        # Fallback
        zone_data = self._stub_nodes.get(zone_id, {})
        open_incidents = [
            node for node in self._stub_nodes.values()
            if node.get("type") == NODE_INCIDENT and not node.get("resolved", True)
            and zone_id.replace("ZONE-", "") in node.get("location", "")
        ]

        return {
            "zone": zone_data,
            "open_incidents": len(open_incidents),
            "required_regulations": [
                self._stub_nodes.get("REG-ISO-3691", {}),
                self._stub_nodes.get("REG-OSHA-178", {}),
            ],
            "compliance_score": 0.87 if not open_incidents else 0.65,
            "last_audit": "2026-06-01",
        }

    async def record_incident(
        self,
        incident_id: str,
        incident_type: str,
        severity: str,
        location: str,
        description: str,
        robots_involved: Optional[List[str]] = None,
        regulations_violated: Optional[List[str]] = None,
    ) -> bool:
        """Record a new safety incident in the knowledge graph."""
        props: Dict[str, Any] = {
            "incident_type": incident_type,
            "severity": severity,
            "location": location,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
        }

        success = await self.create_node(NODE_INCIDENT, incident_id, props)

        if success:
            for robot_id in (robots_involved or []):
                await self.create_relationship(incident_id, REL_INVOLVES, robot_id)

            for reg_id in (regulations_violated or []):
                await self.create_relationship(incident_id, REL_VIOLATED, reg_id)

        return success

    def is_available(self) -> bool:
        """Return True if Neo4j is available and connected."""
        return self._available

    def __repr__(self) -> str:
        backend = "Neo4j" if self._available else "stub"
        return f"WarehouseKnowledgeGraph(backend={backend!r}, uri={self.uri!r})"
