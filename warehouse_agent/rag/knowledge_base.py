"""
Warehouse Knowledge Base using ChromaDB for RAG.

Indexes safety regulations (OSHA, ISO 3691), warehouse layout documents,
and equipment manuals. Provides semantic retrieval for augmenting agent context.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not available; WarehouseKnowledgeBase in stub mode.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Seed knowledge documents
# ---------------------------------------------------------------------------

SEED_DOCUMENTS: List[Dict[str, Any]] = [
    # OSHA Regulations
    {
        "id": "osha-1910-178-l",
        "category": "regulation",
        "source": "OSHA",
        "title": "OSHA 1910.178(l) — Powered Industrial Truck Operator Training",
        "content": (
            "Employers must ensure that each powered industrial truck operator is competent to "
            "operate a powered industrial truck safely, as demonstrated by the successful completion "
            "of the training and evaluation specified in this paragraph. Training program content: "
            "Truck-related topics (operating instructions, warnings and precautions, differences in "
            "types of trucks, controls and instrumentation), Workplace-related topics (surface conditions, "
            "load manipulation, pedestrian traffic, narrow aisles, hazardous locations, ramps). "
            "Evaluation: The employer shall evaluate each operator's performance at least once every "
            "three years. Refresher training is required when an operator is involved in an accident or "
            "near-miss, when an operator is observed operating the vehicle in an unsafe manner."
        ),
        "tags": ["forklift", "training", "operator", "certification", "OSHA"],
    },
    {
        "id": "osha-1910-176",
        "category": "regulation",
        "source": "OSHA",
        "title": "OSHA 1910.176 — Material Handling and Storage",
        "content": (
            "Use of mechanical equipment. Where mechanical handling equipment is used, sufficient "
            "safe clearances shall be allowed for aisles, at loading docks, through doorways and "
            "wherever turns or passage must be made. Aisles and passageways shall be kept clear "
            "and in good repairs, with no obstruction across or in aisles that could create a hazard. "
            "Permanent aisles and passageways shall be appropriately marked. Storage: Stored materials "
            "shall be stacked, blocked, interlocked and otherwise secured to prevent sliding, falling "
            "or collapse. Maximum safe load limits of floors shall be marked on floor plates and posted. "
            "Housekeeping: Storage areas shall be kept free from accumulation of materials that constitute "
            "hazards from tripping, fire, explosion, or pest harborage."
        ),
        "tags": ["storage", "aisles", "housekeeping", "material handling", "OSHA"],
    },
    {
        "id": "osha-1910-157",
        "category": "regulation",
        "source": "OSHA",
        "title": "OSHA 1910.157 — Portable Fire Extinguishers",
        "content": (
            "The employer shall provide portable fire extinguishers and shall mount, locate and "
            "identify them so that they are readily accessible to employees without subjecting the "
            "employees to possible injury. Portable fire extinguishers for use on Class A fires shall "
            "be located so that the travel distance to any extinguisher is 75 feet (22.9 m) or less. "
            "Portable fire extinguishers for use on Class B fires shall be located so that the travel "
            "distance from the Class B hazard area to any extinguisher is 50 feet (15.2 m) or less. "
            "Inspection, maintenance and testing: The employer shall be responsible for the inspection, "
            "maintenance and testing of all portable fire extinguishing equipment. Monthly visual "
            "inspections are required. Annual maintenance checks by a qualified person are required."
        ),
        "tags": ["fire", "extinguisher", "emergency", "safety", "OSHA"],
    },
    # ISO 3691
    {
        "id": "iso-3691-4-2020",
        "category": "regulation",
        "source": "ISO_3691",
        "title": "ISO 3691-4:2020 — Industrial Trucks: Driverless Industrial Trucks and Systems",
        "content": (
            "This document specifies safety requirements and verification of driverless industrial "
            "trucks and systems. Key requirements include: Minimum safety clearance of 500mm (0.5m) "
            "around the robot's planned travel path. Maximum speed in pedestrian zones: 1.2 m/s. "
            "The driverless truck shall be equipped with means to detect obstacles and to stop or "
            "reduce speed. Emergency stop devices shall be accessible and conspicuously marked. "
            "Visual and auditory warning signals are required when the truck is in motion. "
            "The truck shall revert to safe state on communication loss within 500ms. "
            "Zones with pedestrian traffic shall have reduced speed limits and enhanced detection. "
            "The system shall perform daily automatic self-diagnostics before commencing operation. "
            "Battery health monitoring with alert at 20% capacity remaining."
        ),
        "tags": ["AMR", "autonomous", "driverless", "robot", "ISO", "safety", "pedestrian"],
    },
    {
        "id": "iso-3691-2-2016",
        "category": "regulation",
        "source": "ISO_3691",
        "title": "ISO 3691-2:2016 — Industrial Trucks: Additional Requirements for Reach Trucks",
        "content": (
            "This document specifies additional safety requirements and verifications for reach trucks "
            "and straddle trucks. Maximum rated capacity labeling requirements. Stability testing "
            "requirements on slopes and with maximum loads. Operator restraint systems required for "
            "counterbalanced rider trucks with load capacities exceeding 1000 kg. Overhead guard "
            "requirements: must withstand a load of not less than the rated load on the forks "
            "without permanent deformation. Visibility aids required when forward visibility is obscured."
        ),
        "tags": ["reach truck", "forklift", "stability", "capacity", "ISO"],
    },
    # Equipment manuals
    {
        "id": "amr-001-manual",
        "category": "equipment_manual",
        "source": "INTERNAL",
        "title": "AMR Series 4 — Operator and Maintenance Manual",
        "content": (
            "AMR Series 4 autonomous mobile robot specifications: Maximum speed 2.5 m/s (open), "
            "1.0 m/s (pedestrian zones). Payload capacity: 500 kg. Battery: 48V LiFePO4, 80Ah. "
            "Charging time: 2 hours to full (fast charge to 80% in 45 minutes). Operating range: "
            "~8 hours continuous. Navigation: LiDAR SLAM with 360° obstacle detection. "
            "Safety zone: 1.5m warning zone, 0.5m protective zone (triggers emergency stop). "
            "Daily maintenance: Check wheel condition, clean LiDAR lenses, verify e-stop function. "
            "Weekly: Inspect brake pads, check all connector integrity, review error logs. "
            "Monthly: Calibrate load sensors, full LiDAR calibration, battery health test. "
            "Common fault codes: E001=LiDAR obstruction, E002=Low battery, E003=Motor overcurrent, "
            "E004=Communication timeout, E005=Emergency stop activated."
        ),
        "tags": ["AMR", "specifications", "maintenance", "charging", "fault codes", "LiDAR"],
    },
    {
        "id": "forklift-001-manual",
        "category": "equipment_manual",
        "source": "INTERNAL",
        "title": "Autonomous Forklift Series 2 — Technical Manual",
        "content": (
            "Autonomous forklift Series 2 specifications: Lift capacity 2500 kg at 500mm load center. "
            "Maximum lift height: 6.0 m. Maximum travel speed: 1.8 m/s laden, 2.2 m/s unladen. "
            "Battery: 80V lead-acid, 930Ah. Charging: 8-hour overnight charge recommended. "
            "Navigation: Multi-sensor fusion (LiDAR + stereo camera + ultrasonic). "
            "Fork positioning accuracy: ±5mm. Pallet detection: AI vision system with 99.2% accuracy. "
            "Safety features: Forward and rear collision avoidance, lateral protection screens, "
            "hydraulic overload protection, tip-over prevention. "
            "Maintenance: Daily pre-shift inspection required (hydraulics, forks, battery, tires). "
            "Service interval: 250 operating hours. Fault code reference: F001=Hydraulic pressure low, "
            "F002=Fork sensor failure, F003=Steering encoder fault, F004=Navigation map outdated."
        ),
        "tags": ["forklift", "autonomous", "lift capacity", "maintenance", "hydraulics", "specifications"],
    },
    # Warehouse layout
    {
        "id": "warehouse-layout-overview",
        "category": "layout",
        "source": "INTERNAL",
        "title": "Main Warehouse Layout Overview",
        "content": (
            "Warehouse layout overview: Total area 12,500 m². "
            "Zone A (Receiving & Fast-Pick): 2,500 m², nearest to dock doors 1-4. "
            "Houses top-100 fastest-moving SKUs. 2 charging stations (CS-01, CS-02). "
            "Zone B (Bulk Storage): 5,000 m², racking up to 6m height. "
            "Standard pallet storage, reach truck access. 1 charging station (CS-03). "
            "Zone C (Cold Storage): 1,500 m², temperature maintained at 2-8°C. "
            "Refrigerated products only. AMR access restricted (cold-rated AMRs only). "
            "Zone D (Outbound Staging): 2,000 m², dock doors 5-10. "
            "Sortation conveyor system, packing stations 1-8. "
            "Zone E (Maintenance & Charging Hub): 1,500 m², robot maintenance bays 1-6, "
            "4 fast-charging stations (CS-04 to CS-07). "
            "Traffic rules: One-way robot traffic in Zone B aisles. "
            "Pedestrian-only corridors marked in yellow. Robot speed limit 1.0 m/s in Zone A."
        ),
        "tags": ["layout", "zones", "dock", "traffic", "charging", "cold storage"],
    },
    {
        "id": "warehouse-emergency-procedures",
        "category": "safety_procedure",
        "source": "INTERNAL",
        "title": "Warehouse Emergency Response Procedures",
        "content": (
            "Emergency procedures for warehouse operations: "
            "FIRE: 1) Activate nearest fire alarm pull station. 2) Call 911. 3) Use fire extinguisher "
            "only if fire is small and egress is clear (PASS: Pull, Aim, Squeeze, Sweep). "
            "4) Evacuate via marked emergency exits. 5) All robots automatically halt on fire alarm. "
            "ROBOT MALFUNCTION: 1) Press nearest e-stop button. 2) Maintain 3m clearance. "
            "3) Notify fleet manager on radio channel 3. 4) Do not attempt manual intervention. "
            "INJURY: 1) Call first aid team on radio channel 1. 2) Do not move injured person. "
            "3) Keep area clear. 4) Report to HSE within 15 minutes. "
            "CHEMICAL SPILL: 1) Evacuate 10m radius. 2) Identify substance via SDS binder at dock office. "
            "3) Notify HSE and environmental team. 4) Do not attempt cleanup without proper PPE. "
            "POWER OUTAGE: 1) All robots return to base on UPS power. 2) Emergency lighting activates. "
            "3) Manual operations only until power restored. "
            "EVACUATION ASSEMBLY POINT: Parking lot B, minimum 50m from building."
        ),
        "tags": ["emergency", "fire", "evacuation", "first aid", "robot malfunction", "procedures"],
    },
]


# ---------------------------------------------------------------------------
# WarehouseKnowledgeBase implementation
# ---------------------------------------------------------------------------


class KnowledgeChunk(object):
    """A retrieved knowledge chunk with source metadata."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        source: str,
        category: str,
        title: str,
        relevance_score: float,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.category = category
        self.title = title
        self.relevance_score = relevance_score
        self.tags = tags or []

    def to_context_string(self) -> str:
        """Format as a context string for prompt injection."""
        return (
            f"[{self.source} | {self.category.upper()} | Score: {self.relevance_score:.2f}]\n"
            f"Title: {self.title}\n"
            f"Content: {self.content}"
        )


class WarehouseKnowledgeBase:
    """
    RAG knowledge base for warehouse safety regulations and operational documents.

    Backed by ChromaDB with sentence-transformer embeddings.
    Pre-seeded with OSHA, ISO 3691, equipment manuals, and layout documents.
    """

    COLLECTION_NAME = "warehouse_knowledge_base"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        auto_seed: bool = True,
    ) -> None:
        self.persist_directory = persist_directory or "/tmp/warehouse_knowledge_base"
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        self.auto_seed = auto_seed

        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._in_memory_docs: Dict[str, Dict[str, Any]] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Set up ChromaDB and embedding model, then seed documents."""
        if CHROMADB_AVAILABLE:
            try:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "WarehouseKnowledgeBase: ChromaDB collection ready",
                    extra={"collection": self.COLLECTION_NAME},
                )
            except Exception as exc:
                logger.error("ChromaDB init failed: %s", exc)

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as exc:
                logger.error("Sentence transformer load failed: %s", exc)

        if self.auto_seed:
            self._seed_documents()

    def _seed_documents(self) -> None:
        """Load seed documents if collection is empty."""
        current_count = self._get_count()
        if current_count == 0:
            logger.info("Seeding knowledge base with %d documents", len(SEED_DOCUMENTS))
            for doc in SEED_DOCUMENTS:
                self._add_document_sync(doc)
        else:
            logger.info("Knowledge base already has %d documents, skipping seed", current_count)

    def _get_count(self) -> int:
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._in_memory_docs)

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector."""
        if self._embedder is None:
            return None
        try:
            embedding = self._embedder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return None

    def _add_document_sync(self, doc: Dict[str, Any]) -> None:
        """Add a document synchronously (used during initialization)."""
        doc_id = doc.get("id", str(uuid.uuid4()))
        content = doc.get("content", "")
        embed_text = f"{doc.get('title', '')} {content}"

        if self._collection is not None:
            try:
                metadata: Dict[str, Any] = {
                    "source": doc.get("source", "UNKNOWN"),
                    "category": doc.get("category", "general"),
                    "title": doc.get("title", ""),
                    "tags": json.dumps(doc.get("tags", [])),
                }
                embedding = self._embed(embed_text)
                if embedding:
                    self._collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[embed_text],
                        metadatas=[metadata],
                    )
                else:
                    self._collection.add(
                        ids=[doc_id],
                        documents=[embed_text],
                        metadatas=[metadata],
                    )
            except Exception as exc:
                logger.error("Document add failed (%s): %s", doc_id, exc)
                self._in_memory_docs[doc_id] = doc
        else:
            self._in_memory_docs[doc_id] = doc

    async def add_document(
        self,
        title: str,
        content: str,
        source: str = "INTERNAL",
        category: str = "general",
        tags: Optional[List[str]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a new document to the knowledge base."""
        final_id = doc_id or str(uuid.uuid4())
        doc: Dict[str, Any] = {
            "id": final_id,
            "title": title,
            "content": content,
            "source": source,
            "category": category,
            "tags": tags or [],
        }
        self._add_document_sync(doc)
        return final_id

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        min_score: float = 0.2,
    ) -> List[KnowledgeChunk]:
        """
        Retrieve relevant knowledge chunks for a query.

        Args:
            query: Natural language query
            n_results: Maximum number of results
            source_filter: Filter by source (OSHA, ISO_3691, INTERNAL)
            category_filter: Filter by category (regulation, equipment_manual, layout, ...)
            min_score: Minimum similarity score threshold

        Returns:
            List of KnowledgeChunk objects sorted by relevance
        """
        if self._collection is not None:
            try:
                where_conditions: List[Dict[str, Any]] = []
                if source_filter:
                    where_conditions.append({"source": source_filter})
                if category_filter:
                    where_conditions.append({"category": category_filter})

                where: Optional[Dict[str, Any]] = None
                if len(where_conditions) == 1:
                    where = where_conditions[0]
                elif len(where_conditions) > 1:
                    where = {"$and": where_conditions}

                count = self._collection.count()
                if count == 0:
                    return []

                n = min(n_results, count)
                query_kwargs: Dict[str, Any] = {
                    "n_results": n,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    query_kwargs["where"] = where

                embedding = self._embed(query)
                if embedding:
                    query_kwargs["query_embeddings"] = [embedding]
                else:
                    query_kwargs["query_texts"] = [query]

                results = self._collection.query(**query_kwargs)
                chunks: List[KnowledgeChunk] = []

                if results and results.get("ids"):
                    ids = results["ids"][0]
                    distances = results.get("distances", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]
                    documents = results.get("documents", [[]])[0]

                    for i, chunk_id in enumerate(ids):
                        distance = distances[i] if distances else 1.0
                        similarity = 1.0 - (distance / 2.0)

                        if similarity < min_score:
                            continue

                        meta = metadatas[i] if metadatas else {}
                        tags_raw = meta.get("tags", "[]")
                        try:
                            tags_list = json.loads(tags_raw)
                        except (json.JSONDecodeError, TypeError):
                            tags_list = []

                        chunk = KnowledgeChunk(
                            chunk_id=chunk_id,
                            content=documents[i] if documents else "",
                            source=meta.get("source", "UNKNOWN"),
                            category=meta.get("category", "general"),
                            title=meta.get("title", ""),
                            relevance_score=similarity,
                            tags=tags_list,
                        )
                        chunks.append(chunk)

                return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

            except Exception as exc:
                logger.error("Knowledge base retrieval failed: %s", exc)

        # Fallback: simple keyword matching
        return self._fallback_retrieve(query, n_results, source_filter, category_filter)

    def _fallback_retrieve(
        self,
        query: str,
        n_results: int,
        source_filter: Optional[str],
        category_filter: Optional[str],
    ) -> List[KnowledgeChunk]:
        """Keyword-based fallback retrieval."""
        query_words = set(query.lower().split())
        results: List[Tuple[float, KnowledgeChunk]] = []

        for doc_id, doc in self._in_memory_docs.items():
            if source_filter and doc.get("source") != source_filter:
                continue
            if category_filter and doc.get("category") != category_filter:
                continue

            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                chunk = KnowledgeChunk(
                    chunk_id=doc_id,
                    content=doc.get("content", ""),
                    source=doc.get("source", "UNKNOWN"),
                    category=doc.get("category", "general"),
                    title=doc.get("title", ""),
                    relevance_score=score,
                    tags=doc.get("tags", []),
                )
                results.append((score, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:n_results]]

    async def build_rag_context(
        self,
        query: str,
        n_results: int = 3,
        max_chars: int = 3000,
    ) -> str:
        """
        Build a RAG context string for injection into the agent's system prompt.

        Retrieves relevant documents and formats them with source attribution.
        """
        chunks = await self.retrieve(query, n_results=n_results)

        if not chunks:
            return ""

        lines: List[str] = ["=== KNOWLEDGE BASE CONTEXT ==="]
        total_chars = 0

        for chunk in chunks:
            entry = chunk.to_context_string()
            if total_chars + len(entry) > max_chars:
                # Truncate to fit
                remaining = max_chars - total_chars
                if remaining > 100:
                    lines.append(entry[:remaining] + "...[truncated]")
                break
            lines.append(entry)
            lines.append("---")
            total_chars += len(entry)

        lines.append("==============================")
        return "\n".join(lines)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge base (metadata only)."""
        if self._collection is not None:
            try:
                results = self._collection.get(include=["metadatas"])
                if results and results.get("ids"):
                    docs = []
                    for i, doc_id in enumerate(results["ids"]):
                        meta = results["metadatas"][i] if results.get("metadatas") else {}
                        docs.append({"id": doc_id, **meta})
                    return docs
            except Exception as exc:
                logger.error("List documents failed: %s", exc)

        return [
            {"id": doc_id, **{k: v for k, v in doc.items() if k != "content"}}
            for doc_id, doc in self._in_memory_docs.items()
        ]

    def __repr__(self) -> str:
        backend = "ChromaDB" if self._collection is not None else "in-memory"
        return f"WarehouseKnowledgeBase(backend={backend!r}, documents={self._get_count()})"
