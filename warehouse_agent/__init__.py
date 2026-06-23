"""
warehouse_agent — WarehouseGPT LLM Agent Package (Phase 6)

Main exports:
- WarehouseAgent: Core agent class with Claude Sonnet 4.6, tool use loop, memory, RAG
- AgentResponse: Structured agent response model
- EpisodicMemory: ChromaDB-backed episodic memory
- WorkingMemory: Sliding window working memory with incident buffer
- WarehouseKnowledgeBase: RAG knowledge base (OSHA, ISO 3691, equipment manuals)
- WarehouseKnowledgeGraph: Neo4j knowledge graph for causal chain analysis
- AgentEvaluator: Accuracy, latency, and tool-use evaluation suite
- app: FastAPI application instance
"""

from warehouse_agent.agent import AgentResponse, WarehouseAgent
from warehouse_agent.api.main import app
from warehouse_agent.evaluation.eval import AgentEvaluator, DEFAULT_EVAL_SUITE, EvalCase
from warehouse_agent.memory.episodic import EpisodicMemory
from warehouse_agent.memory.working import WorkingMemory
from warehouse_agent.rag.knowledge_base import WarehouseKnowledgeBase
from warehouse_agent.rag.knowledge_graph import WarehouseKnowledgeGraph
from warehouse_agent.tools.warehouse_tools import TOOL_HANDLERS, TOOLS

__all__ = [
    "WarehouseAgent",
    "AgentResponse",
    "EpisodicMemory",
    "WorkingMemory",
    "WarehouseKnowledgeBase",
    "WarehouseKnowledgeGraph",
    "AgentEvaluator",
    "EvalCase",
    "DEFAULT_EVAL_SUITE",
    "TOOLS",
    "TOOL_HANDLERS",
    "app",
]

__version__ = "1.0.0"
