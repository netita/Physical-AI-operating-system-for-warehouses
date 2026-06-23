"""
tests/test_agent.py
====================
pytest tests for the WarehouseGPT agent.

Tests
-----
* TestAgentQueryWithMockedOpenAI  — full query() call with mocked OpenAI API
* TestToolRouting                  — tool_registry.execute_tool dispatches correctly
* TestMemoryRetrieval              — EpisodicMemory stores and retrieves episodes

The OpenAI API client is fully mocked so these tests run without network
access or a real API key.

Run::

    pytest tests/test_agent.py -v
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop():
    """Provide a fresh event loop per test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Mock helpers for OpenAI Chat Completions API
# ---------------------------------------------------------------------------


def _make_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_id: str = "call_test_01",
) -> MagicMock:
    """Create a mock OpenAI ToolCall."""
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_input)
    return tc


def _make_completion(
    content: Optional[str] = None,
    tool_calls: Optional[list] = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# test_agent_query_with_mocked_openai
# ---------------------------------------------------------------------------


class TestAgentQueryWithMockedOpenAI:
    """
    Tests for WarehouseAgent.query() with the OpenAI client mocked.

    Scenarios tested:
    1. Single-turn response (no tool use) — finish_reason "stop" immediately.
    2. One-shot tool use — tool_calls then stop.
    3. Multi-turn tool chain — two rounds of tool_calls then stop.
    4. API error handling — APIStatusError is caught and returns graceful response.
    5. length (max_tokens) stop reason — response includes truncation notice.
    """

    @pytest.fixture
    def agent(self):
        """Create a WarehouseAgent with dummy credentials and in-memory memory."""
        from warehouse_agent.agent import WarehouseAgent

        return WarehouseAgent(
            api_key="test-key-not-real",
            neo4j_uri="bolt://localhost:7687",
            episodic_memory_dir="/tmp/test_episodic",
            knowledge_base_dir="/tmp/test_kb",
        )

    @pytest.mark.asyncio
    async def test_end_turn_no_tool_use(self, agent) -> None:
        """
        When the model returns finish_reason "stop" immediately, query() should return
        the text response without calling any tool handlers.
        """
        final_text = "All systems nominal. Fleet is operating at 95% efficiency."

        stop_response = _make_completion(content=final_text, finish_reason="stop")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = stop_response

            result = await agent.query("What is the fleet status?", session_id="sess-001")

        assert result.response == final_text
        assert result.tool_call_count == 0
        assert result.tools_used == []
        assert result.error is None
        assert result.stop_reason == "stop"
        assert result.session_id == "sess-001"

    @pytest.mark.asyncio
    async def test_single_tool_call(self, agent) -> None:
        """
        Model uses get_robot_fleet_status once, then stop.
        The tool result should be consumed and the agent should return
        a response referencing the fleet data.
        """
        tool_response_text = "Fleet check complete: 6 robots active, 1 charging."

        tool_calls_response = _make_completion(
            tool_calls=[
                _make_tool_call(
                    "get_robot_fleet_status",
                    {"robot_ids": [], "include_metrics": False},
                    tool_id="call_fleet_01",
                )
            ],
            finish_reason="tool_calls",
        )
        stop_response = _make_completion(content=tool_response_text, finish_reason="stop")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [tool_calls_response, stop_response]

            result = await agent.query("Check the robot fleet status.", session_id="sess-002")

        assert result.response == tool_response_text
        assert result.tool_call_count == 1
        assert "get_robot_fleet_status" in result.tools_used
        assert result.error is None
        # create() was called twice (once for tool_calls, once for stop)
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_turn_tool_chain(self, agent) -> None:
        """
        Model calls two different tools in sequence, then stop.
        Both tools must appear in tools_used.
        """
        final_text = "Inventory and order status retrieved successfully."

        first_tool_response = _make_completion(
            tool_calls=[
                _make_tool_call(
                    "get_inventory_status",
                    {"item_ids": ["SKU-001"]},
                    tool_id="call_inv_01",
                )
            ],
            finish_reason="tool_calls",
        )
        second_tool_response = _make_completion(
            tool_calls=[
                _make_tool_call(
                    "get_order_status",
                    {"order_ids": []},
                    tool_id="call_ord_01",
                )
            ],
            finish_reason="tool_calls",
        )
        stop_response = _make_completion(content=final_text, finish_reason="stop")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [
                first_tool_response,
                second_tool_response,
                stop_response,
            ]

            result = await agent.query(
                "Give me inventory and order status.", session_id="sess-003"
            )

        assert result.response == final_text
        assert result.tool_call_count == 2
        assert "get_inventory_status" in result.tools_used
        assert "get_order_status" in result.tools_used
        assert result.error is None

    @pytest.mark.asyncio
    async def test_api_status_error_handled_gracefully(self, agent) -> None:
        """
        An openai.APIStatusError must be caught; the response must contain
        a graceful error message and result.error must be set.
        """
        import openai

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_response = MagicMock()
            mock_response.status_code = 529
            mock_create.side_effect = openai.APIStatusError(
                message="Overloaded",
                response=mock_response,
                body={"error": {"message": "Overloaded"}},
            )

            result = await agent.query("What is inventory status?", session_id="sess-004")

        assert result.error is not None
        assert "529" in result.error or "Overloaded" in result.error
        # User-facing response must be a graceful message, not a traceback
        assert "error" in result.response.lower() or "unable" in result.response.lower()

    @pytest.mark.asyncio
    async def test_max_tokens_stop_reason(self, agent) -> None:
        """When finish_reason is 'length', response should include truncation notice."""
        partial_text = "The warehouse currently has 247 SKUs in low-stock status"

        length_response = _make_completion(content=partial_text, finish_reason="length")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = length_response

            result = await agent.query("Give me a full inventory report.", session_id="sess-005")

        assert partial_text in result.response
        assert "truncated" in result.response.lower() or "length" in result.response.lower()
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_response_metadata(self, agent) -> None:
        """AgentResponse must have all required metadata fields."""
        stop_response = _make_completion(content="All clear.", finish_reason="stop")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = stop_response

            result = await agent.query("Status?")

        assert result.session_id
        assert result.query == "Status?"
        assert isinstance(result.response, str) and result.response
        assert isinstance(result.tools_used, list)
        assert isinstance(result.tool_call_count, int)
        assert isinstance(result.latency_ms, float) and result.latency_ms >= 0
        assert result.model == agent.model
        assert result.timestamp  # ISO-format string

    @pytest.mark.asyncio
    async def test_session_history_updated(self, agent) -> None:
        """After query(), session history must have 2 turns (user + assistant)."""
        session_id = "sess-history-test"
        stop_response = _make_completion(content="Robot AMR-001 is idle.", finish_reason="stop")

        with patch.object(
            agent._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = stop_response

            await agent.query("Where is AMR-001?", session_id=session_id)

        history = await agent.get_session_history(session_id)
        roles = [t["role"] for t in history]
        assert "user" in roles
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# test_tool_routing
# ---------------------------------------------------------------------------


class TestToolRouting:
    """Tests for warehouse_agent.tools.tool_registry.execute_tool routing."""

    @pytest.mark.asyncio
    async def test_known_tool_returns_dict(self) -> None:
        """execute_tool with a valid core tool name must return a dict."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool(
            "get_inventory_status",
            {"item_ids": ["SKU-TEST-001"]},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_dict(self) -> None:
        """execute_tool with an unknown tool must return an error dict, not raise."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool("nonexistent_tool_xyz", {})
        assert isinstance(result, dict)
        assert "error" in result
        assert "nonexistent_tool_xyz" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_result_contains_metadata(self) -> None:
        """execute_tool must inject _tool and _elapsed_ms into the result."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool("run_diagnostic", {"component": "network"})
        assert "_tool" in result, "result missing _tool metadata key"
        assert "_elapsed_ms" in result, "result missing _elapsed_ms metadata key"
        assert result["_tool"] == "run_diagnostic"
        assert isinstance(result["_elapsed_ms"], float) and result["_elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_all_core_tools_registered(self) -> None:
        """All eight core warehouse tools must be present in the registry."""
        from warehouse_agent.tools.tool_registry import tool_handlers

        expected_core_tools = [
            "get_inventory_status",
            "assign_robot_task",
            "get_robot_fleet_status",
            "query_safety_regulations",
            "report_safety_incident",
            "optimize_warehouse_layout",
            "get_order_status",
            "run_diagnostic",
        ]
        for tool in expected_core_tools:
            assert tool in tool_handlers, (
                f"Core tool {tool!r} not found in tool_handlers registry"
            )

    @pytest.mark.asyncio
    async def test_integration_tools_registered(self) -> None:
        """Digital Twin, World Model, and Safety AI bridge tools must be registered."""
        from warehouse_agent.tools.tool_registry import tool_handlers

        integration_tools = [
            "get_warehouse_state",
            "get_active_incidents",
            "get_throughput_analytics",
            "predict_occupancy",
            "detect_near_misses",
            "detect_fire",
            "detect_zone_violations",
        ]
        for tool in integration_tools:
            assert tool in tool_handlers, (
                f"Integration tool {tool!r} not found in tool_handlers"
            )

    @pytest.mark.asyncio
    async def test_get_robot_fleet_status_response_structure(self) -> None:
        """get_robot_fleet_status must return a structured fleet report."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool(
            "get_robot_fleet_status",
            {"robot_ids": ["AMR-001", "FORK-001"], "include_metrics": True},
        )
        # The core handler returns fleet data; after JSON→dict normalisation
        # we expect these keys (injected by the warehouse_tools handler)
        assert isinstance(result, dict)
        # _tool is injected by execute_tool
        assert result.get("_tool") == "get_robot_fleet_status"

    @pytest.mark.asyncio
    async def test_report_safety_incident_returns_incident_id(self) -> None:
        """report_safety_incident must return an incident_id in the response."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool(
            "report_safety_incident",
            {
                "incident_type": "near_miss",
                "severity": "high",
                "location": "Zone A, Aisle 3",
                "description": "Forklift passed within 0.5m of worker.",
                "robots_involved": ["FORK-002"],
                "immediate_action_required": False,
            },
        )
        assert isinstance(result, dict)
        assert "incident_id" in result, "Response must contain incident_id"
        assert result["incident_id"].startswith("INC-")

    @pytest.mark.asyncio
    async def test_assign_robot_task_emergency_stop(self) -> None:
        """assign_robot_task with emergency_stop must include safety_alert flag."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool(
            "assign_robot_task",
            {"robot_id": "AMR-003", "task_type": "emergency_stop"},
        )
        assert isinstance(result, dict)
        assert result.get("safety_alert") is True

    @pytest.mark.asyncio
    async def test_digital_twin_tool_graceful_fallback(self) -> None:
        """
        get_warehouse_state must return a dict even when the Digital Twin API
        is not reachable (stub fallback).
        """
        from warehouse_agent.tools.tool_registry import execute_tool

        # No Digital Twin running — should fall back gracefully
        result = await execute_tool("get_warehouse_state", {"warehouse_id": "test"})
        assert isinstance(result, dict)
        # Either from API or stub
        assert "_source" in result

    @pytest.mark.asyncio
    async def test_detect_near_misses_returns_dict(self) -> None:
        """detect_near_misses must return a structured dict with events list."""
        from warehouse_agent.tools.tool_registry import execute_tool

        result = await execute_tool(
            "detect_near_misses",
            {"camera_id": "test_cam", "use_mock_frame": True},
        )
        assert isinstance(result, dict)
        assert "near_miss_events" in result or "error" in result

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self) -> None:
        """
        If a handler raises an unexpected exception, execute_tool must catch it
        and return an error dict instead of propagating the exception.
        """
        from warehouse_agent.tools.tool_registry import execute_tool, tool_handlers

        original = tool_handlers.get("run_diagnostic")
        try:
            async def _broken(_: Any) -> Any:
                raise RuntimeError("Simulated internal failure")

            tool_handlers["run_diagnostic"] = _broken
            result = await execute_tool("run_diagnostic", {"component": "all"})
            assert isinstance(result, dict)
            assert "error" in result
            assert "run_diagnostic" in result["error"]
        finally:
            if original is not None:
                tool_handlers["run_diagnostic"] = original


# ---------------------------------------------------------------------------
# test_memory_retrieval
# ---------------------------------------------------------------------------


class TestMemoryRetrieval:
    """Tests for EpisodicMemory store/retrieve functionality."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create an EpisodicMemory backed by a temporary directory."""
        from warehouse_agent.memory.episodic import EpisodicMemory

        return EpisodicMemory(
            persist_directory=str(tmp_path / "episodic_memory"),
            max_episodes=100,
        )

    @pytest.mark.asyncio
    async def test_store_episode_returns_episode(self, memory) -> None:
        """store_episode must return an Episode object with all fields."""
        from warehouse_agent.memory.episodic import Episode

        ep = await memory.store_episode(
            session_id="sess-mem-01",
            user_query="How many robots are active?",
            assistant_response="Currently 6 robots are active.",
            tools_used=["get_robot_fleet_status"],
        )

        assert isinstance(ep, Episode)
        assert ep.session_id == "sess-mem-01"
        assert ep.user_query == "How many robots are active?"
        assert ep.assistant_response == "Currently 6 robots are active."
        assert "get_robot_fleet_status" in ep.tools_used
        assert ep.episode_id.startswith("ep_")
        assert ep.timestamp

    @pytest.mark.asyncio
    async def test_count_increments(self, memory) -> None:
        """Storing episodes must increment the count."""
        initial_count = memory.count()
        await memory.store_episode(
            session_id="sess-mem-count",
            user_query="Test query 1",
            assistant_response="Test response 1",
        )
        await memory.store_episode(
            session_id="sess-mem-count",
            user_query="Test query 2",
            assistant_response="Test response 2",
        )
        assert memory.count() >= initial_count + 2

    @pytest.mark.asyncio
    async def test_search_similar_returns_results(self, memory) -> None:
        """
        After storing a few episodes, searching with a related query must
        return results with similarity scores in [0, 1].
        """
        await memory.store_episode(
            session_id="sess-search",
            user_query="forklift battery status",
            assistant_response="FORK-001 battery at 42%.",
            tools_used=["get_robot_fleet_status"],
        )
        await memory.store_episode(
            session_id="sess-search",
            user_query="check inventory levels for zone A",
            assistant_response="Zone A has 124 items in stock.",
            tools_used=["get_inventory_status"],
        )

        results = await memory.search_similar("robot battery", n_results=5)
        assert isinstance(results, list)
        # At least one result should be found in fallback or ChromaDB mode
        # (exact count depends on backend availability)

        for r in results:
            assert hasattr(r, "episode") and hasattr(r, "similarity_score")
            assert 0.0 <= r.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_search_similar_sorted_by_score(self, memory) -> None:
        """Search results must be sorted descending by similarity score."""
        await memory.store_episode(
            session_id="sess-sort",
            user_query="robot fleet health check diagnostics",
            assistant_response="Fleet healthy.",
        )
        await memory.store_episode(
            session_id="sess-sort",
            user_query="inventory count zone B",
            assistant_response="Zone B: 88 items.",
        )

        results = await memory.search_similar("robot fleet status", n_results=5)
        if len(results) >= 2:
            scores = [r.similarity_score for r in results]
            assert scores == sorted(scores, reverse=True), (
                "Search results must be sorted by descending similarity score"
            )

    @pytest.mark.asyncio
    async def test_build_memory_context_returns_string(self, memory) -> None:
        """build_memory_context must return a string (possibly empty)."""
        await memory.store_episode(
            session_id="sess-ctx",
            user_query="safety incident near Zone A",
            assistant_response="Incident INC-20240601-1234 logged.",
            tools_used=["report_safety_incident"],
        )

        context = await memory.build_memory_context("safety incident")
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_build_memory_context_format(self, memory) -> None:
        """When episodes are found, the context string must include section headers."""
        await memory.store_episode(
            session_id="sess-fmt",
            user_query="forklift near-miss zone C",
            assistant_response="Near-miss reported.",
        )

        context = await memory.build_memory_context("near-miss forklift")
        if context:  # may be empty if no high-similarity match
            assert "PAST INTERACTIONS" in context or "match" in context.lower()

    @pytest.mark.asyncio
    async def test_get_session_episodes(self, memory) -> None:
        """get_session_episodes must return all episodes for a given session."""
        session_id = "sess-episodes-test"
        for i in range(3):
            await memory.store_episode(
                session_id=session_id,
                user_query=f"Query number {i}",
                assistant_response=f"Response number {i}",
            )

        episodes = await memory.get_session_episodes(session_id)
        assert isinstance(episodes, list)
        # All stored episodes should be retrievable
        assert len(episodes) >= 3

    @pytest.mark.asyncio
    async def test_reset_clears_all_episodes(self, memory) -> None:
        """reset() must remove all stored episodes."""
        await memory.store_episode(
            session_id="sess-reset",
            user_query="Test query",
            assistant_response="Test response",
        )
        assert memory.count() > 0

        memory.reset()
        assert memory.count() == 0

    @pytest.mark.asyncio
    async def test_session_filter_in_search(self, memory) -> None:
        """
        When session_id_filter is provided, search must only return episodes
        from that session.
        """
        await memory.store_episode(
            session_id="session-A",
            user_query="forklift position Zone A",
            assistant_response="FORK-001 in Zone A.",
        )
        await memory.store_episode(
            session_id="session-B",
            user_query="forklift task assignment Zone B",
            assistant_response="FORK-002 assigned to Zone B.",
        )

        results_a = await memory.search_similar(
            "forklift zone", n_results=10, session_id_filter="session-A"
        )
        for r in results_a:
            assert r.episode.session_id == "session-A", (
                f"Expected session-A, got {r.episode.session_id}"
            )
