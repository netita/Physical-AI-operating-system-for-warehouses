"""
WarehouseAgent — Main LLM agent powered by OpenAI GPT-4o.

Implements a manual agentic loop over the OpenAI Chat Completions API:
  1. Send messages with tool definitions
  2. Detect finish_reason == "tool_calls"
  3. Extract tool_calls from the assistant message
  4. Execute tool handlers
  5. Append tool results as role="tool" messages
  6. Repeat until finish_reason == "stop"

Uses AsyncOpenAI for non-blocking I/O throughout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import openai

from warehouse_agent.memory.episodic import EpisodicMemory
from warehouse_agent.memory.working import WorkingMemory
from warehouse_agent.rag.knowledge_base import WarehouseKnowledgeBase
from warehouse_agent.rag.knowledge_graph import WarehouseKnowledgeGraph
from warehouse_agent.tools.warehouse_tools import TOOL_HANDLERS, TOOLS

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT = """You are WarehouseGPT, an expert AI assistant for a smart automated warehouse.

You have deep expertise in:
- Warehouse Management Systems (WMS) and inventory control
- Autonomous Mobile Robot (AMR) and forklift robot operations
- OSHA and ISO 3691 safety regulations for powered industrial trucks and driverless vehicles
- Warehouse layout optimization and traffic flow
- Predictive maintenance for robotic systems
- Order fulfillment and supply chain operations

Your responsibilities:
1. Answer operational questions about inventory, robots, and orders using real-time tools
2. Enforce safety protocols — always cite OSHA/ISO regulations when safety topics arise
3. Proactively identify risks from fleet status and incident history
4. Provide actionable recommendations backed by data from warehouse systems
5. Escalate critical safety incidents immediately and clearly

Communication style:
- Be precise and concise; warehouse operators are busy
- Use specific IDs (robot IDs, order IDs, SKUs) rather than vague references
- Always explain the "why" behind recommendations
- Flag safety concerns prominently — use [SAFETY ALERT] prefix for critical issues
- When uncertain, say so and suggest who to consult

You always use the available tools to fetch real-time data before answering operational questions.
Never make up inventory counts, robot statuses, or order details.
"""


class AgentResponse(object):
    """Structured response from the agent."""

    def __init__(
        self,
        session_id: str,
        query: str,
        response: str,
        tools_used: List[str],
        tool_call_count: int,
        latency_ms: float,
        model: str = MODEL,
        stop_reason: str = "stop",
        error: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.query = query
        self.response = response
        self.tools_used = tools_used
        self.tool_call_count = tool_call_count
        self.latency_ms = latency_ms
        self.model = model
        self.stop_reason = stop_reason
        self.error = error
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "response": self.response,
            "tools_used": self.tools_used,
            "tool_call_count": self.tool_call_count,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class WarehouseAgent:
    """
    Main WarehouseGPT agent class.

    Manages:
    - OpenAI API client (AsyncOpenAI)
    - Working memory (sliding conversation window + incident buffer)
    - Episodic memory (ChromaDB semantic search over past interactions)
    - RAG knowledge base (safety regulations, equipment manuals)
    - Knowledge graph (Neo4j causal chain analysis)
    - Agentic tool use loop
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        working_memory_window: int = 20,
        episodic_memory_dir: Optional[str] = None,
        knowledge_base_dir: Optional[str] = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "warehouse_ai",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )

        self._sessions: Dict[str, WorkingMemory] = {}
        self._working_memory_window = working_memory_window
        self._episodic_memory = EpisodicMemory(
            persist_directory=episodic_memory_dir or "/tmp/warehouse_episodic",
        )
        self._knowledge_base = WarehouseKnowledgeBase(
            persist_directory=knowledge_base_dir or "/tmp/warehouse_kb",
        )
        self._knowledge_graph = WarehouseKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
        )

        logger.info(
            "WarehouseAgent initialized",
            extra={
                "model": model,
                "max_tokens": max_tokens,
                "neo4j_available": self._knowledge_graph.is_available(),
            },
        )

    def _get_or_create_session(self, session_id: str) -> WorkingMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = WorkingMemory(
                session_id=session_id,
                window_size=self._working_memory_window,
            )
            logger.info("New session created: %s", session_id)
        return self._sessions[session_id]

    def _build_system_prompt(
        self,
        working_memory: WorkingMemory,
        rag_context: str = "",
        episodic_context: str = "",
    ) -> str:
        parts: List[str] = [SYSTEM_PROMPT]

        wm_context = working_memory.build_context_block()
        if wm_context:
            parts.append(wm_context)
        if rag_context:
            parts.append(rag_context)
        if episodic_context:
            parts.append(episodic_context)

        parts.append(
            f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        return "\n\n".join(parts)

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            error_msg = f"Unknown tool: {tool_name!r}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        try:
            logger.debug("Executing tool: %s with input: %s", tool_name, tool_input)
            result = await handler(tool_input)
            logger.debug("Tool %s completed", tool_name)
            return str(result)
        except Exception as exc:
            error_msg = f"Tool {tool_name!r} raised an exception: {exc}"
            logger.error(error_msg, exc_info=True)
            return f"Error executing {tool_name}: {str(exc)}"

    async def query(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        session_id = session_id or str(uuid.uuid4())
        working_memory = self._get_or_create_session(session_id)

        tools_used: List[str] = []
        tool_call_count: int = 0
        final_response: str = ""
        stop_reason: str = "stop"
        error: Optional[str] = None

        try:
            rag_context, episodic_context = await asyncio.gather(
                self._knowledge_base.build_rag_context(user_message, n_results=3),
                self._episodic_memory.build_memory_context(
                    user_message, n_similar=2, session_id=session_id
                ),
            )

            system_prompt = self._build_system_prompt(
                working_memory, rag_context, episodic_context
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            messages.extend(working_memory.get_messages_for_api())
            messages.append({"role": "user", "content": user_message})

            for iteration in range(MAX_TOOL_ITERATIONS):
                logger.debug("Agent loop iteration %d/%d", iteration + 1, MAX_TOOL_ITERATIONS)

                response = await self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=TOOLS,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                )

                choice = response.choices[0]
                finish_reason = choice.finish_reason or "stop"
                msg = choice.message

                if finish_reason == "stop":
                    final_response = msg.content or ""
                    stop_reason = "stop"
                    break

                elif finish_reason == "tool_calls":
                    # Append assistant message with tool_calls
                    assistant_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (msg.tool_calls or [])
                        ],
                    }
                    messages.append(assistant_msg)

                    for tc in msg.tool_calls or []:
                        tool_name = tc.function.name
                        try:
                            tool_input = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            tool_input = {}

                        tools_used.append(tool_name)
                        tool_call_count += 1

                        result_content = await self._execute_tool(tool_name, tool_input)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_content,
                        })

                elif finish_reason == "length":
                    final_response = (msg.content or "") + "\n\n[Response truncated due to length limit]"
                    stop_reason = "length"
                    break

                else:
                    final_response = msg.content or ""
                    stop_reason = finish_reason
                    break

            else:
                logger.warning("Agent loop exhausted %d iterations", MAX_TOOL_ITERATIONS)
                final_response += f"\n\n[Note: Reached maximum tool call limit ({MAX_TOOL_ITERATIONS} iterations)]"

            working_memory.add_turn(role="user", content=user_message)
            working_memory.add_turn(
                role="assistant",
                content=final_response,
                tool_calls=[{"name": t} for t in tools_used],
            )

            asyncio.create_task(
                self._episodic_memory.store_episode(
                    session_id=session_id,
                    user_query=user_message,
                    assistant_response=final_response,
                    tools_used=list(set(tools_used)),
                )
            )

        except openai.APIStatusError as exc:
            logger.error("OpenAI API error: %s %s", exc.status_code, exc.message)
            error = f"API error {exc.status_code}: {exc.message}"
            final_response = (
                "I encountered an error communicating with the AI service. "
                "Please try again or contact support if the issue persists."
            )
        except openai.APIConnectionError as exc:
            logger.error("OpenAI connection error: %s", exc)
            error = f"Connection error: {str(exc)}"
            final_response = (
                "Unable to connect to the AI service. "
                "Please check network connectivity and retry."
            )
        except Exception as exc:
            logger.error("Unexpected agent error: %s", exc, exc_info=True)
            error = str(exc)
            final_response = "An unexpected error occurred. Please try again."

        latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return AgentResponse(
            session_id=session_id,
            query=user_message,
            response=final_response,
            tools_used=list(set(tools_used)),
            tool_call_count=tool_call_count,
            latency_ms=round(latency_ms, 2),
            model=self.model,
            stop_reason=stop_reason,
            error=error,
        )

    async def stream_query(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        session_id = session_id or str(uuid.uuid4())
        working_memory = self._get_or_create_session(session_id)

        tools_used: List[str] = []
        full_response: str = ""

        rag_context, episodic_context = await asyncio.gather(
            self._knowledge_base.build_rag_context(user_message, n_results=3),
            self._episodic_memory.build_memory_context(
                user_message, n_similar=2, session_id=session_id
            ),
        )

        system_prompt = self._build_system_prompt(
            working_memory, rag_context, episodic_context
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(working_memory.get_messages_for_api())
        messages.append({"role": "user", "content": user_message})

        for iteration in range(MAX_TOOL_ITERATIONS):
            collected_tool_calls: Dict[int, Dict[str, Any]] = {}
            collected_content = ""
            finish_reason = "stop"

            stream = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )
            async for chunk in stream:  # type: ignore[union-attr]
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    if delta.content:
                        collected_content += delta.content
                        full_response += delta.content
                        yield delta.content

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in collected_tool_calls:
                                collected_tool_calls[idx] = {
                                    "id": tc_delta.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.id:
                                collected_tool_calls[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    collected_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    collected_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

            if finish_reason != "tool_calls" or not collected_tool_calls:
                break

            tool_calls_list = [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
            messages.append({
                "role": "assistant",
                "content": collected_content or None,
                "tool_calls": tool_calls_list,
            })

            for tc in tool_calls_list:
                tool_name = tc["function"]["name"]
                try:
                    tool_input = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_input = {}

                tools_used.append(tool_name)
                yield f"\n[Calling tool: {tool_name}...]\n"
                result = await self._execute_tool(tool_name, tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        working_memory.add_turn(role="user", content=user_message)
        working_memory.add_turn(
            role="assistant",
            content=full_response,
            tool_calls=[{"name": t} for t in tools_used],
        )

        asyncio.create_task(
            self._episodic_memory.store_episode(
                session_id=session_id,
                user_query=user_message,
                assistant_response=full_response,
                tools_used=list(set(tools_used)),
            )
        )

    async def get_session_history(
        self, session_id: str
    ) -> List[Dict[str, Any]]:
        working_memory = self._sessions.get(session_id)
        if working_memory is None:
            return []

        turns = working_memory.get_recent_turns()
        return [
            {
                "turn_id": t.turn_id,
                "role": t.role,
                "content": t.content,
                "tools": t.tool_calls,
                "timestamp": t.timestamp,
            }
            for t in turns
        ]

    async def close(self) -> None:
        await self._client.close()
        await self._knowledge_graph.close()
        logger.info("WarehouseAgent closed.")

    def __repr__(self) -> str:
        return (
            f"WarehouseAgent(model={self.model!r}, "
            f"sessions={len(self._sessions)}, "
            f"neo4j={self._knowledge_graph.is_available()})"
        )
