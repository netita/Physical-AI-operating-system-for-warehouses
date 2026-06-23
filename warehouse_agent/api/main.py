"""
WarehouseGPT FastAPI application.

Endpoints:
- POST /query          — Synchronous query with full agent response
- GET  /history/{sid}  — Conversation history for a session
- WebSocket /chat      — Bidirectional WebSocket chat
- GET  /query/stream   — Server-Sent Events (SSE) streaming response
- GET  /health         — Health check
- GET  /sessions       — List active sessions
- DELETE /sessions/{sid} — Clear a session
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from warehouse_agent.agent import WarehouseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan — single shared agent instance
# ---------------------------------------------------------------------------

_agent: Optional[WarehouseAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and teardown the shared WarehouseAgent."""
    global _agent
    logger.info("Starting WarehouseGPT API...")

    _agent = WarehouseAgent(
        api_key=os.environ.get("OPENAI_API_KEY"),
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "warehouse_ai"),
        episodic_memory_dir=os.environ.get("EPISODIC_MEMORY_DIR"),
        knowledge_base_dir=os.environ.get("KNOWLEDGE_BASE_DIR"),
    )
    logger.info("WarehouseAgent initialized: %r", _agent)

    yield

    if _agent is not None:
        await _agent.close()
        logger.info("WarehouseAgent closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WarehouseGPT API",
    description="LLM-powered intelligent assistant for warehouse operations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> WarehouseAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    message: str = Field(..., min_length=1, max_length=4000, description="User query")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation continuity"
    )


class QueryResponse(BaseModel):
    """Response body from POST /query."""

    session_id: str
    query: str
    response: str
    tools_used: List[str]
    tool_call_count: int
    latency_ms: float
    model: str
    stop_reason: str
    timestamp: str
    error: Optional[str] = None


class HistoryTurn(BaseModel):
    """A single conversation turn in history."""

    turn_id: int
    role: str
    content: str
    tools: List[Dict[str, Any]]
    timestamp: str


class HistoryResponse(BaseModel):
    """Response body from GET /history/{session_id}."""

    session_id: str
    turn_count: int
    turns: List[HistoryTurn]


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    agent_ready: bool
    neo4j_available: bool
    model: str


class StreamQueryRequest(BaseModel):
    """Query params for SSE streaming."""

    message: str = Field(..., description="User query")
    session_id: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui() -> HTMLResponse:
    """Serve the WarehouseGPT chat UI."""
    import pathlib
    html_path = pathlib.Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    agent = _agent
    return HealthResponse(
        status="ok" if agent is not None else "starting",
        agent_ready=agent is not None,
        neo4j_available=agent._knowledge_graph.is_available() if agent else False,
        model=agent.model if agent else "unknown",
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Send a synchronous query to the WarehouseGPT agent.

    The agent will use tools as needed and return a complete response.
    For streaming, use GET /query/stream instead.
    """
    agent = get_agent()
    session_id = request.session_id or str(uuid.uuid4())

    result = await agent.query(
        user_message=request.message,
        session_id=session_id,
    )

    return QueryResponse(
        session_id=result.session_id,
        query=result.query,
        response=result.response,
        tools_used=result.tools_used,
        tool_call_count=result.tool_call_count,
        latency_ms=result.latency_ms,
        model=result.model,
        stop_reason=result.stop_reason,
        timestamp=result.timestamp,
        error=result.error,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["Agent"])
async def get_history(session_id: str) -> HistoryResponse:
    """
    Retrieve conversation history for a session.
    """
    agent = get_agent()
    turns_raw = await agent.get_session_history(session_id)

    if not turns_raw and session_id not in agent._sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found",
        )

    turns = [
        HistoryTurn(
            turn_id=t["turn_id"],
            role=t["role"],
            content=t["content"],
            tools=t.get("tools", []),
            timestamp=t["timestamp"],
        )
        for t in turns_raw
    ]

    return HistoryResponse(
        session_id=session_id,
        turn_count=len(turns),
        turns=turns,
    )


@app.get("/query/stream", tags=["Agent"])
async def stream_query(
    message: str = Query(..., min_length=1, max_length=4000),
    session_id: Optional[str] = Query(default=None),
) -> StreamingResponse:
    """
    Stream a response from the WarehouseGPT agent using Server-Sent Events (SSE).

    Client receives events:
    - data: {"type": "text", "content": "..."}
    - data: {"type": "tool_call", "name": "..."}
    - data: {"type": "done", "session_id": "..."}
    - data: {"type": "error", "message": "..."}
    """
    agent = get_agent()
    final_session_id = session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in agent.stream_query(
                user_message=message,
                session_id=final_session_id,
            ):
                if chunk.startswith("\n[Calling tool:"):
                    # Tool call notification
                    tool_name = chunk.strip().replace("[Calling tool: ", "").replace("...]", "").strip()
                    event_data = json.dumps({"type": "tool_call", "name": tool_name})
                else:
                    event_data = json.dumps({"type": "text", "content": chunk})

                yield f"data: {event_data}\n\n"

            # Send done event
            done_data = json.dumps({"type": "done", "session_id": final_session_id})
            yield f"data: {done_data}\n\n"

        except Exception as exc:
            logger.error("SSE stream error: %s", exc, exc_info=True)
            error_data = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for bidirectional real-time chat.

    Client sends: {"message": "...", "session_id": "optional"}
    Server sends: {"type": "chunk"|"done"|"error", "content": "...", "session_id": "..."}
    """
    await websocket.accept()
    agent = get_agent()
    session_id: Optional[str] = None

    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "content": "Invalid JSON payload"}
                )
                continue

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json(
                    {"type": "error", "content": "Message cannot be empty"}
                )
                continue

            # Use provided session_id or maintain one per connection
            req_session_id = data.get("session_id")
            if req_session_id:
                session_id = req_session_id
            elif session_id is None:
                session_id = str(uuid.uuid4())

            # Stream response back over WebSocket
            try:
                async for chunk in agent.stream_query(
                    user_message=message,
                    session_id=session_id,
                ):
                    await websocket.send_json(
                        {"type": "chunk", "content": chunk, "session_id": session_id}
                    )

                await websocket.send_json(
                    {"type": "done", "content": "", "session_id": session_id}
                )

            except Exception as exc:
                logger.error("WebSocket agent error: %s", exc, exc_info=True)
                await websocket.send_json(
                    {"type": "error", "content": f"Agent error: {str(exc)}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (session: %s)", session_id)
    except Exception as exc:
        logger.error("WebSocket unexpected error: %s", exc, exc_info=True)
        try:
            await websocket.send_json(
                {"type": "error", "content": "Connection error — please reconnect"}
            )
        except Exception:
            pass


@app.get("/sessions", tags=["Operations"])
async def list_sessions() -> Dict[str, Any]:
    """List all active session IDs and their turn counts."""
    agent = get_agent()
    sessions_info = {}
    for sid, wm in agent._sessions.items():
        state = wm.get_state()
        sessions_info[sid] = {
            "turn_count": state.turn_count,
            "active_incidents": len(state.active_incidents),
            "window_used": state.turn_count,
        }

    return {
        "active_sessions": len(sessions_info),
        "sessions": sessions_info,
    }


@app.delete("/sessions/{session_id}", tags=["Operations"])
async def clear_session(session_id: str) -> Dict[str, Any]:
    """Clear a session's working memory."""
    agent = get_agent()

    if session_id not in agent._sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found",
        )

    agent._sessions[session_id].clear()
    del agent._sessions[session_id]

    return {"status": "cleared", "session_id": session_id}


@app.get("/knowledge-base/documents", tags=["Knowledge"])
async def list_knowledge_base_documents() -> Dict[str, Any]:
    """List all documents in the knowledge base."""
    agent = get_agent()
    docs = agent._knowledge_base.list_documents()
    return {"document_count": len(docs), "documents": docs}


@app.post("/knowledge-base/documents", tags=["Knowledge"])
async def add_knowledge_base_document(
    title: str,
    content: str,
    source: str = "INTERNAL",
    category: str = "general",
) -> Dict[str, Any]:
    """Add a new document to the RAG knowledge base."""
    agent = get_agent()
    doc_id = await agent._knowledge_base.add_document(
        title=title,
        content=content,
        source=source,
        category=category,
    )
    return {"status": "added", "document_id": doc_id, "title": title}
