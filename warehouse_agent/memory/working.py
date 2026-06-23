"""
Working memory module for WarehouseGPT.

Implements a sliding window of conversation turns with an incident context buffer.
Provides structured context injection for the agent's system prompt.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """A single conversation turn (user message + optional assistant response)."""

    turn_id: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    role: str  # "user" | "assistant"
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentContext(BaseModel):
    """Active incident tracked in working memory."""

    incident_id: str
    type: str
    severity: str
    location: str
    description: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    resolution_notes: Optional[str] = None


class WorkingMemoryState(BaseModel):
    """Snapshot of working memory state."""

    session_id: str
    turn_count: int
    window_size: int
    active_incidents: List[IncidentContext]
    recent_tool_calls: List[str]
    current_context_summary: Optional[str]
    robot_fleet_snapshot: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Working memory implementation
# ---------------------------------------------------------------------------


class WorkingMemory:
    """
    Sliding window working memory for conversation context.

    Maintains:
    - A sliding window of recent conversation turns (configurable size)
    - An incident context buffer for active safety incidents
    - A snapshot of robot fleet state
    - Recent tool call history for deduplication
    """

    def __init__(
        self,
        session_id: str,
        window_size: int = 20,
        max_incidents: int = 10,
    ) -> None:
        self.session_id = session_id
        self.window_size = window_size
        self.max_incidents = max_incidents

        self._turns: Deque[ConversationTurn] = deque(maxlen=window_size)
        self._incidents: Deque[IncidentContext] = deque(maxlen=max_incidents)
        self._recent_tool_calls: Deque[str] = deque(maxlen=50)
        self._turn_counter: int = 0
        self._robot_fleet_snapshot: Optional[Dict[str, Any]] = None
        self._context_summary: Optional[str] = None

        logger.info(
            "WorkingMemory initialized",
            extra={"session_id": session_id, "window_size": window_size},
        )

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a conversation turn to working memory."""
        self._turn_counter += 1
        turn = ConversationTurn(
            turn_id=self._turn_counter,
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            metadata=metadata or {},
        )
        self._turns.append(turn)

        # Track tool calls for deduplication
        for tc in (tool_calls or []):
            tool_name = tc.get("name", "unknown")
            self._recent_tool_calls.append(tool_name)

        return turn

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get the n most recent turns (or all if n is None)."""
        turns = list(self._turns)
        if n is not None:
            turns = turns[-n:]
        return turns

    def get_messages_for_api(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Format recent turns as Anthropic API messages list.

        Returns a list of {"role": ..., "content": ...} dicts suitable for
        passing directly to the messages API.
        """
        turns = self.get_recent_turns(n)
        messages: List[Dict[str, Any]] = []

        for turn in turns:
            if turn.content:
                messages.append({"role": turn.role, "content": turn.content})

        return messages

    # ------------------------------------------------------------------
    # Incident context buffer
    # ------------------------------------------------------------------

    def add_incident(
        self,
        incident_id: str,
        incident_type: str,
        severity: str,
        location: str,
        description: str,
    ) -> IncidentContext:
        """Add an active incident to the context buffer."""
        incident = IncidentContext(
            incident_id=incident_id,
            type=incident_type,
            severity=severity,
            location=location,
            description=description,
        )
        self._incidents.append(incident)
        logger.warning(
            "Incident added to working memory",
            extra={"incident_id": incident_id, "severity": severity},
        )
        return incident

    def resolve_incident(self, incident_id: str, resolution_notes: str = "") -> bool:
        """Mark an incident as resolved."""
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                incident.resolved = True
                incident.resolution_notes = resolution_notes
                logger.info("Incident resolved", extra={"incident_id": incident_id})
                return True
        return False

    def get_active_incidents(self) -> List[IncidentContext]:
        """Return list of unresolved incidents."""
        return [inc for inc in self._incidents if not inc.resolved]

    def has_critical_incidents(self) -> bool:
        """Check if there are any unresolved critical incidents."""
        return any(
            inc.severity == "critical" and not inc.resolved
            for inc in self._incidents
        )

    # ------------------------------------------------------------------
    # Fleet snapshot cache
    # ------------------------------------------------------------------

    def update_fleet_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Cache the latest robot fleet state."""
        self._robot_fleet_snapshot = {
            "data": snapshot,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_fleet_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return cached fleet snapshot or None if not set."""
        return self._robot_fleet_snapshot

    # ------------------------------------------------------------------
    # Context summary
    # ------------------------------------------------------------------

    def set_context_summary(self, summary: str) -> None:
        """Set a condensed summary of the current session context."""
        self._context_summary = summary

    def get_context_summary(self) -> Optional[str]:
        """Return the current context summary."""
        return self._context_summary

    # ------------------------------------------------------------------
    # System prompt context injection
    # ------------------------------------------------------------------

    def build_context_block(self) -> str:
        """
        Build a structured context block for injection into the system prompt.

        This gives the agent immediate awareness of active incidents, recent
        tool usage, and other session state without needing to query memory.
        """
        lines: List[str] = ["=== WORKING MEMORY CONTEXT ==="]

        # Active incidents
        active = self.get_active_incidents()
        if active:
            lines.append(f"\nACTIVE INCIDENTS ({len(active)}):")
            for inc in active:
                severity_flag = " [CRITICAL]" if inc.severity == "critical" else ""
                lines.append(
                    f"  - [{inc.incident_id}]{severity_flag} {inc.type} at {inc.location}: {inc.description}"
                )
        else:
            lines.append("\nACTIVE INCIDENTS: None")

        # Recent tool usage summary
        if self._recent_tool_calls:
            recent_unique = list(dict.fromkeys(list(self._recent_tool_calls)[-10:]))
            lines.append(f"\nRECENT TOOL CALLS: {', '.join(recent_unique)}")

        # Fleet snapshot summary
        if self._robot_fleet_snapshot:
            snap = self._robot_fleet_snapshot
            cached_at = snap.get("cached_at", "unknown")
            fleet_data = snap.get("data", {})
            fleet_size = fleet_data.get("fleet_size", "?")
            active_robots = fleet_data.get("active_robots", "?")
            lines.append(
                f"\nFLEET SNAPSHOT (cached {cached_at}): {active_robots}/{fleet_size} robots active"
            )

        # Context summary
        if self._context_summary:
            lines.append(f"\nSESSION SUMMARY: {self._context_summary}")

        # Turn count
        lines.append(f"\nCONVERSATION TURNS: {self._turn_counter} (window: {len(self._turns)}/{self.window_size})")
        lines.append("=" * 30)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> WorkingMemoryState:
        """Return a serializable snapshot of the current working memory state."""
        return WorkingMemoryState(
            session_id=self.session_id,
            turn_count=self._turn_counter,
            window_size=self.window_size,
            active_incidents=self.get_active_incidents(),
            recent_tool_calls=list(self._recent_tool_calls)[-20:],
            current_context_summary=self._context_summary,
            robot_fleet_snapshot=self._robot_fleet_snapshot,
        )

    def clear(self) -> None:
        """Reset working memory (new session)."""
        self._turns.clear()
        self._incidents.clear()
        self._recent_tool_calls.clear()
        self._turn_counter = 0
        self._robot_fleet_snapshot = None
        self._context_summary = None
        logger.info("WorkingMemory cleared", extra={"session_id": self.session_id})

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(session_id={self.session_id!r}, "
            f"turns={self._turn_counter}, "
            f"window={len(self._turns)}/{self.window_size}, "
            f"active_incidents={len(self.get_active_incidents())})"
        )
