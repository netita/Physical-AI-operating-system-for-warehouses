"""
Episodic memory module for WarehouseGPT.

Persists conversation episodes in ChromaDB with sentence-transformer embeddings.
Supports semantic retrieval of similar past interactions and auto-summarization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (ChromaDB + sentence-transformers)
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed; EpisodicMemory will operate in stub mode.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed; embeddings will be disabled.")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Episode(object):
    """Represents a stored episodic memory entry."""

    def __init__(
        self,
        episode_id: str,
        session_id: str,
        user_query: str,
        assistant_response: str,
        tools_used: List[str],
        timestamp: str,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.episode_id = episode_id
        self.session_id = session_id
        self.user_query = user_query
        self.assistant_response = assistant_response
        self.tools_used = tools_used
        self.timestamp = timestamp
        self.summary = summary
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "tools_used": self.tools_used,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            episode_id=data["episode_id"],
            session_id=data["session_id"],
            user_query=data["user_query"],
            assistant_response=data["assistant_response"],
            tools_used=data.get("tools_used", []),
            timestamp=data["timestamp"],
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
        )


class EpisodicSearchResult(object):
    """A semantic search result from episodic memory."""

    def __init__(self, episode: Episode, similarity_score: float) -> None:
        self.episode = episode
        self.similarity_score = similarity_score

    def __repr__(self) -> str:
        return (
            f"EpisodicSearchResult(score={self.similarity_score:.3f}, "
            f"episode_id={self.episode.episode_id!r})"
        )


# ---------------------------------------------------------------------------
# EpisodicMemory implementation
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """
    Persistent episodic memory backed by ChromaDB.

    Stores conversation episodes with semantic embeddings for retrieval.
    Falls back to in-memory storage when ChromaDB is unavailable.
    """

    COLLECTION_NAME = "warehouse_episodic_memory"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        max_episodes: int = 10_000,
        auto_summarize_threshold: int = 5,
    ) -> None:
        self.persist_directory = persist_directory or "/tmp/warehouse_episodic_memory"
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        self.max_episodes = max_episodes
        self.auto_summarize_threshold = auto_summarize_threshold

        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._fallback_store: Dict[str, Episode] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Set up ChromaDB client and embedding model."""
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
                    "EpisodicMemory: ChromaDB collection ready",
                    extra={"collection": self.COLLECTION_NAME, "path": self.persist_directory},
                )
            except Exception as exc:
                logger.error("ChromaDB initialization failed, using fallback: %s", exc)
                self._client = None
                self._collection = None
        else:
            logger.info("EpisodicMemory: using in-memory fallback (no ChromaDB)")

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info("EpisodicMemory: embedding model loaded (%s)", self.embedding_model_name)
            except Exception as exc:
                logger.error("Embedding model load failed: %s", exc)
                self._embedder = None

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text string."""
        if self._embedder is None:
            return None
        try:
            embedding = self._embedder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return None

    def _make_episode_id(self, session_id: str, query: str) -> str:
        """Generate a deterministic but unique episode ID."""
        content = f"{session_id}:{query}:{uuid.uuid4()}"
        return "ep_" + hashlib.sha256(content.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def store_episode(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        tools_used: Optional[List[str]] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Store a conversation episode in episodic memory.

        The episode is embedded and indexed for semantic search.
        """
        episode_id = self._make_episode_id(session_id, user_query)
        timestamp = datetime.now(timezone.utc).isoformat()

        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            user_query=user_query,
            assistant_response=assistant_response,
            tools_used=tools_used or [],
            timestamp=timestamp,
            summary=summary,
            metadata=metadata or {},
        )

        # Text to embed: query + summary (if available) for richer semantics
        embed_text = user_query
        if summary:
            embed_text = f"{user_query}\n{summary}"

        if self._collection is not None:
            try:
                chroma_metadata: Dict[str, Any] = {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "tools_used": json.dumps(tools_used or []),
                    "has_summary": summary is not None,
                }
                if metadata:
                    # ChromaDB metadata values must be str/int/float/bool
                    for k, v in metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            chroma_metadata[k] = v

                embedding = self._embed(embed_text)
                if embedding:
                    self._collection.add(
                        ids=[episode_id],
                        embeddings=[embedding],
                        documents=[embed_text],
                        metadatas=[chroma_metadata],
                    )
                else:
                    # Fall back to document-only storage (ChromaDB will use its own embedder)
                    self._collection.add(
                        ids=[episode_id],
                        documents=[embed_text],
                        metadatas=[chroma_metadata],
                    )
                logger.debug("Episode stored in ChromaDB: %s", episode_id)
            except Exception as exc:
                logger.error("ChromaDB store failed, using fallback: %s", exc)
                self._fallback_store[episode_id] = episode
        else:
            self._fallback_store[episode_id] = episode

        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        session_id_filter: Optional[str] = None,
        min_score: float = 0.3,
    ) -> List[EpisodicSearchResult]:
        """
        Semantic search over episodic memory.

        Returns episodes most similar to the query, sorted by similarity score.
        """
        if self._collection is not None:
            try:
                where: Optional[Dict[str, Any]] = None
                if session_id_filter:
                    where = {"session_id": session_id_filter}

                embedding = self._embed(query)
                query_kwargs: Dict[str, Any] = {
                    "n_results": min(n_results, self._collection.count() or 1),
                    "include": ["documents", "metadatas", "distances"],
                }
                if embedding:
                    query_kwargs["query_embeddings"] = [embedding]
                else:
                    query_kwargs["query_texts"] = [query]
                if where:
                    query_kwargs["where"] = where

                results = self._collection.query(**query_kwargs)

                episodes: List[EpisodicSearchResult] = []
                if results and results.get("ids"):
                    ids = results["ids"][0]
                    distances = results.get("distances", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]
                    documents = results.get("documents", [[]])[0]

                    for i, ep_id in enumerate(ids):
                        # ChromaDB cosine distance: 0 = identical, 2 = opposite
                        # Convert to similarity: 1 - (distance / 2)
                        distance = distances[i] if distances else 1.0
                        similarity = 1.0 - (distance / 2.0)

                        if similarity < min_score:
                            continue

                        meta = metadatas[i] if metadatas else {}
                        tools = json.loads(meta.get("tools_used", "[]"))
                        ep = Episode(
                            episode_id=ep_id,
                            session_id=meta.get("session_id", ""),
                            user_query=documents[i] if documents else "",
                            assistant_response="[stored in ChromaDB — fetch by ID for full response]",
                            tools_used=tools,
                            timestamp=meta.get("timestamp", ""),
                            metadata=meta,
                        )
                        episodes.append(EpisodicSearchResult(episode=ep, similarity_score=similarity))

                return sorted(episodes, key=lambda r: r.similarity_score, reverse=True)

            except Exception as exc:
                logger.error("ChromaDB search failed: %s", exc)

        # Fallback: simple keyword matching
        return self._fallback_search(query, n_results, session_id_filter)

    def _fallback_search(
        self,
        query: str,
        n_results: int,
        session_id_filter: Optional[str],
    ) -> List[EpisodicSearchResult]:
        """Simple keyword-based search for fallback mode."""
        query_lower = query.lower()
        results: List[EpisodicSearchResult] = []

        for episode in self._fallback_store.values():
            if session_id_filter and episode.session_id != session_id_filter:
                continue

            # Simple word overlap scoring
            episode_text = f"{episode.user_query} {episode.summary or ''}".lower()
            query_words = set(query_lower.split())
            episode_words = set(episode_text.split())
            overlap = len(query_words & episode_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                results.append(EpisodicSearchResult(episode=episode, similarity_score=score))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:n_results]

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID."""
        if episode_id in self._fallback_store:
            return self._fallback_store[episode_id]

        if self._collection is not None:
            try:
                result = self._collection.get(ids=[episode_id], include=["documents", "metadatas"])
                if result and result["ids"]:
                    meta = result["metadatas"][0]
                    return Episode(
                        episode_id=episode_id,
                        session_id=meta.get("session_id", ""),
                        user_query=result["documents"][0],
                        assistant_response="",
                        tools_used=json.loads(meta.get("tools_used", "[]")),
                        timestamp=meta.get("timestamp", ""),
                        metadata=meta,
                    )
            except Exception as exc:
                logger.error("Episode fetch failed: %s", exc)

        return None

    async def get_session_episodes(
        self, session_id: str, limit: int = 50
    ) -> List[Episode]:
        """Return all episodes for a given session, most recent first."""
        if self._collection is not None:
            try:
                results = self._collection.get(
                    where={"session_id": session_id},
                    include=["documents", "metadatas"],
                    limit=limit,
                )
                if results and results["ids"]:
                    episodes: List[Episode] = []
                    for i, ep_id in enumerate(results["ids"]):
                        meta = results["metadatas"][i]
                        ep = Episode(
                            episode_id=ep_id,
                            session_id=session_id,
                            user_query=results["documents"][i],
                            assistant_response="",
                            tools_used=json.loads(meta.get("tools_used", "[]")),
                            timestamp=meta.get("timestamp", ""),
                            metadata=meta,
                        )
                        episodes.append(ep)
                    # Sort by timestamp descending
                    episodes.sort(key=lambda e: e.timestamp, reverse=True)
                    return episodes[:limit]
            except Exception as exc:
                logger.error("Session episodes fetch failed: %s", exc)

        # Fallback
        episodes = [
            ep for ep in self._fallback_store.values() if ep.session_id == session_id
        ]
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]

    # ------------------------------------------------------------------
    # Auto-summarization
    # ------------------------------------------------------------------

    async def build_memory_context(
        self,
        query: str,
        n_similar: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a memory context string for injection into the system prompt.

        Retrieves similar past episodes and formats them as context.
        """
        similar = await self.search_similar(query, n_results=n_similar, session_id_filter=session_id)

        if not similar:
            return ""

        lines: List[str] = ["=== RELEVANT PAST INTERACTIONS ==="]
        for result in similar:
            ep = result.episode
            score_pct = int(result.similarity_score * 100)
            lines.append(
                f"\n[{score_pct}% match | {ep.timestamp[:10]}]"
            )
            lines.append(f"User asked: {ep.user_query[:200]}")
            if ep.summary:
                lines.append(f"Summary: {ep.summary[:300]}")
            if ep.tools_used:
                lines.append(f"Tools used: {', '.join(ep.tools_used)}")

        lines.append("=" * 35)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored episodes."""
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._fallback_store)

    def reset(self) -> None:
        """Delete all episodes (for testing only)."""
        if self._collection is not None:
            try:
                self._client.delete_collection(self.COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.error("Reset failed: %s", exc)
        self._fallback_store.clear()

    def __repr__(self) -> str:
        backend = "ChromaDB" if self._collection is not None else "in-memory"
        return f"EpisodicMemory(backend={backend!r}, episodes={self.count()})"
