"""
AgentEvaluator — Evaluation suite for the WarehouseGPT agent.

Measures:
- Response accuracy (semantic similarity vs. expected answers)
- Latency (p50/p95/p99 percentiles using numpy)
- Tool use precision and recall
- Safety regulation citation accuracy

Uses sentence-transformers for semantic similarity scoring (cosine similarity).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available; latency percentiles will use basic stats.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available; semantic similarity disabled.")


# ---------------------------------------------------------------------------
# Evaluation data models
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single evaluation test case."""

    case_id: str
    query: str
    expected_answer: str  # Reference answer for semantic similarity
    expected_tools: List[str] = field(default_factory=list)  # Tools that should be called
    required_safety_citations: List[str] = field(default_factory=list)  # e.g. ["OSHA", "ISO 3691"]
    category: str = "general"  # general, safety, inventory, fleet, optimization
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from evaluating a single test case."""

    case_id: str
    query: str
    actual_response: str
    actual_tools_used: List[str]
    latency_ms: float

    # Scores
    semantic_similarity: float = 0.0
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0
    safety_citation_score: float = 0.0
    overall_score: float = 0.0

    # Meta
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class EvalSuiteResult:
    """Aggregate results from a full evaluation suite run."""

    suite_name: str
    n_cases: int
    n_errors: int
    run_timestamp: str

    # Accuracy metrics
    mean_semantic_similarity: float = 0.0
    mean_tool_f1: float = 0.0
    mean_safety_citation_score: float = 0.0
    mean_overall_score: float = 0.0

    # Latency metrics (milliseconds)
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Per-category breakdown
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Individual results
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "n_cases": self.n_cases,
            "n_errors": self.n_errors,
            "run_timestamp": self.run_timestamp,
            "accuracy": {
                "mean_semantic_similarity": round(self.mean_semantic_similarity, 4),
                "mean_tool_f1": round(self.mean_tool_f1, 4),
                "mean_safety_citation_score": round(self.mean_safety_citation_score, 4),
                "mean_overall_score": round(self.mean_overall_score, 4),
            },
            "latency_ms": {
                "mean": round(self.mean_latency_ms, 2),
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
                "min": round(self.min_latency_ms, 2),
                "max": round(self.max_latency_ms, 2),
            },
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
        }

    def summary(self) -> str:
        """Return a human-readable summary of evaluation results."""
        lines = [
            f"=== Evaluation Suite: {self.suite_name} ===",
            f"Timestamp: {self.run_timestamp}",
            f"Cases: {self.n_cases} total, {self.n_errors} errors",
            "",
            "Accuracy:",
            f"  Semantic similarity:      {self.mean_semantic_similarity:.3f}",
            f"  Tool use F1:              {self.mean_tool_f1:.3f}",
            f"  Safety citation score:    {self.mean_safety_citation_score:.3f}",
            f"  Overall score:            {self.mean_overall_score:.3f}",
            "",
            "Latency (ms):",
            f"  Mean:  {self.mean_latency_ms:.1f}",
            f"  p50:   {self.p50_latency_ms:.1f}",
            f"  p95:   {self.p95_latency_ms:.1f}",
            f"  p99:   {self.p99_latency_ms:.1f}",
            f"  Min:   {self.min_latency_ms:.1f}",
            f"  Max:   {self.max_latency_ms:.1f}",
        ]
        if self.category_scores:
            lines.append("")
            lines.append("Category Breakdown:")
            for cat, score in sorted(self.category_scores.items()):
                lines.append(f"  {cat:<20} {score:.3f}")
        lines.append("=" * 40)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default evaluation suite
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SUITE: List[EvalCase] = [
    EvalCase(
        case_id="INV-001",
        query="What is the current stock level for SKU-4821?",
        expected_answer="SKU-4821 has low inventory levels near the reorder point and may require restocking soon.",
        expected_tools=["get_inventory_status"],
        category="inventory",
    ),
    EvalCase(
        case_id="INV-002",
        query="Show me all items with low stock in Zone A.",
        expected_answer="There are multiple items in Zone A with stock below reorder points that need attention.",
        expected_tools=["get_inventory_status"],
        category="inventory",
    ),
    EvalCase(
        case_id="FLEET-001",
        query="What is the current status of AMR-003?",
        expected_answer="AMR-003 is currently charging with a low battery level around 22%.",
        expected_tools=["get_robot_fleet_status"],
        category="fleet",
    ),
    EvalCase(
        case_id="FLEET-002",
        query="Which robots are currently available for new tasks?",
        expected_answer="Several AMR robots are idle and available for assignment while others are charging or on active tasks.",
        expected_tools=["get_robot_fleet_status"],
        category="fleet",
    ),
    EvalCase(
        case_id="SAFETY-001",
        query="What is the OSHA requirement for forklift operator training?",
        expected_answer=(
            "OSHA 1910.178(l) requires formal training, practical exercises, and evaluation "
            "every three years. Refresher training is required after accidents or unsafe operation."
        ),
        expected_tools=["query_safety_regulations"],
        required_safety_citations=["OSHA", "1910.178"],
        category="safety",
    ),
    EvalCase(
        case_id="SAFETY-002",
        query="What speed limit applies to robots in pedestrian zones per ISO 3691?",
        expected_answer="ISO 3691-4:2020 requires robots to travel at maximum 1.2 m/s in pedestrian zones.",
        expected_tools=["query_safety_regulations"],
        required_safety_citations=["ISO", "3691", "1.2"],
        category="safety",
    ),
    EvalCase(
        case_id="TASK-001",
        query="Assign AMR-001 to pick from location A-12-3 and deliver to Zone D staging.",
        expected_answer="Task has been successfully assigned to AMR-001 with estimated completion time.",
        expected_tools=["assign_robot_task"],
        category="fleet",
    ),
    EvalCase(
        case_id="INCIDENT-001",
        query="Report a near-miss incident in Zone B aisle 4 involving AMR-002 and a pedestrian.",
        expected_answer="The near-miss incident has been logged, the HSE officer notified, and appropriate escalation completed.",
        expected_tools=["report_safety_incident"],
        required_safety_citations=["incident"],
        category="safety",
    ),
    EvalCase(
        case_id="OPT-001",
        query="Analyze and suggest layout optimizations to improve warehouse throughput.",
        expected_answer="Throughput can be improved by optimizing pick paths, relocating fast-moving SKUs, and implementing bidirectional traffic flow.",
        expected_tools=["optimize_warehouse_layout"],
        category="optimization",
    ),
    EvalCase(
        case_id="DIAG-001",
        query="Run a diagnostic on all charging stations.",
        expected_answer="Charging station diagnostics show most stations are operational but CS-03 has a connection fault requiring maintenance.",
        expected_tools=["run_diagnostic"],
        category="maintenance",
    ),
]


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------


class AgentEvaluator:
    """
    Evaluates WarehouseGPT agent performance on structured test cases.

    Metrics:
    - Semantic similarity: cosine similarity between actual and expected responses
    - Tool precision/recall/F1: how well the agent selects appropriate tools
    - Safety citation score: whether required safety references appear in the response
    - Latency: p50/p95/p99 percentiles across all test cases
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        # Weights for overall score computation
        self.weights = weights or {
            "semantic_similarity": 0.5,
            "tool_f1": 0.3,
            "safety_citation": 0.2,
        }

        self._embedder: Optional[Any] = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info("Evaluator: embedding model loaded (%s)", self.embedding_model_name)
            except Exception as exc:
                logger.error("Evaluator: embedding model load failed: %s", exc)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            a = np.array(vec_a)
            b = np.array(vec_b)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # Pure Python fallback
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = sum(x * x for x in vec_a) ** 0.5
        norm_b = sum(x * x for x in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def compute_semantic_similarity(self, actual: str, expected: str) -> float:
        """
        Compute semantic similarity between actual and expected responses.

        Returns a score in [0, 1]. Falls back to simple word overlap if
        sentence-transformers is unavailable.
        """
        if not actual or not expected:
            return 0.0

        if self._embedder is not None:
            try:
                embeddings = self._embedder.encode(
                    [actual, expected], normalize_embeddings=True
                )
                return float(self._cosine_similarity(
                    embeddings[0].tolist(), embeddings[1].tolist()
                ))
            except Exception as exc:
                logger.error("Semantic similarity embedding failed: %s", exc)

        # Fallback: Jaccard similarity on word tokens
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        intersection = len(actual_words & expected_words)
        union = len(actual_words | expected_words)
        return intersection / union if union > 0 else 0.0

    def compute_tool_metrics(
        self,
        actual_tools: List[str],
        expected_tools: List[str],
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 for tool use.

        Returns (precision, recall, f1).
        """
        if not expected_tools and not actual_tools:
            return 1.0, 1.0, 1.0
        if not expected_tools:
            return 0.0, 1.0, 0.0
        if not actual_tools:
            return 0.0, 0.0, 0.0

        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        tp = len(actual_set & expected_set)
        precision = tp / len(actual_set) if actual_set else 0.0
        recall = tp / len(expected_set) if expected_set else 0.0

        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def compute_safety_citation_score(
        self,
        actual_response: str,
        required_citations: List[str],
    ) -> float:
        """
        Check whether required safety references appear in the actual response.

        Returns fraction of required citations found.
        """
        if not required_citations:
            return 1.0

        response_lower = actual_response.lower()
        found = sum(
            1 for citation in required_citations
            if citation.lower() in response_lower
        )
        return found / len(required_citations)

    def _compute_overall_score(
        self,
        semantic_similarity: float,
        tool_f1: float,
        safety_citation_score: float,
    ) -> float:
        """Compute weighted overall score."""
        w = self.weights
        return (
            w.get("semantic_similarity", 0.5) * semantic_similarity
            + w.get("tool_f1", 0.3) * tool_f1
            + w.get("safety_citation", 0.2) * safety_citation_score
        )

    def _compute_latency_percentiles(
        self, latencies: List[float]
    ) -> Dict[str, float]:
        """Compute latency statistics including p50, p95, p99."""
        if not latencies:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}

        if NUMPY_AVAILABLE:
            arr = np.array(latencies)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        # Pure Python fallback
        sorted_l = sorted(latencies)
        n = len(sorted_l)

        def percentile(p: float) -> float:
            idx = int(p / 100 * n)
            return sorted_l[min(idx, n - 1)]

        return {
            "mean": sum(latencies) / n,
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
            "min": sorted_l[0],
            "max": sorted_l[-1],
        }

    async def evaluate_case(
        self,
        case: EvalCase,
        agent: Any,  # WarehouseAgent — avoid circular import
        session_id: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate a single test case against the agent.

        Args:
            case: The test case to evaluate
            agent: WarehouseAgent instance
            session_id: Optional session ID for isolated evaluation

        Returns:
            EvalResult with all computed metrics
        """
        import uuid as _uuid
        eval_session = session_id or f"eval-{_uuid.uuid4()}"
        error: Optional[str] = None
        actual_response = ""
        actual_tools: List[str] = []
        latency_ms = 0.0

        try:
            start = time.perf_counter()
            agent_response = await agent.query(
                user_message=case.query,
                session_id=eval_session,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            actual_response = agent_response.response
            actual_tools = agent_response.tools_used
            error = agent_response.error

        except Exception as exc:
            error = str(exc)
            latency_ms = 0.0
            logger.error("Evaluation case %s failed: %s", case.case_id, exc)

        # Compute metrics
        semantic_sim = self.compute_semantic_similarity(actual_response, case.expected_answer)
        precision, recall, f1 = self.compute_tool_metrics(actual_tools, case.expected_tools)
        safety_score = self.compute_safety_citation_score(
            actual_response, case.required_safety_citations
        )
        overall = self._compute_overall_score(semantic_sim, f1, safety_score)

        return EvalResult(
            case_id=case.case_id,
            query=case.query,
            actual_response=actual_response,
            actual_tools_used=actual_tools,
            latency_ms=latency_ms,
            semantic_similarity=semantic_sim,
            tool_precision=precision,
            tool_recall=recall,
            tool_f1=f1,
            safety_citation_score=safety_score,
            overall_score=overall,
            error=error,
        )

    async def run_suite(
        self,
        agent: Any,
        cases: Optional[List[EvalCase]] = None,
        suite_name: str = "default",
        concurrency: int = 1,
        session_prefix: str = "eval",
    ) -> EvalSuiteResult:
        """
        Run a full evaluation suite.

        Args:
            agent: WarehouseAgent instance
            cases: Test cases (defaults to DEFAULT_EVAL_SUITE)
            suite_name: Name for the suite run
            concurrency: Number of cases to run in parallel
            session_prefix: Prefix for evaluation session IDs

        Returns:
            EvalSuiteResult with aggregate metrics
        """
        eval_cases = cases or DEFAULT_EVAL_SUITE
        run_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Starting evaluation suite %r with %d cases (concurrency=%d)",
            suite_name, len(eval_cases), concurrency,
        )

        results: List[EvalResult] = []

        if concurrency <= 1:
            # Sequential evaluation
            for i, case in enumerate(eval_cases):
                session_id = f"{session_prefix}-{i}-{case.case_id}"
                logger.info("Evaluating case %d/%d: %s", i + 1, len(eval_cases), case.case_id)
                result = await self.evaluate_case(case, agent, session_id)
                results.append(result)
        else:
            # Parallel evaluation with semaphore
            semaphore = asyncio.Semaphore(concurrency)

            async def _eval_with_semaphore(case: EvalCase, idx: int) -> EvalResult:
                async with semaphore:
                    session_id = f"{session_prefix}-{idx}-{case.case_id}"
                    return await self.evaluate_case(case, agent, session_id)

            tasks = [
                asyncio.create_task(_eval_with_semaphore(case, i))
                for i, case in enumerate(eval_cases)
            ]
            results = list(await asyncio.gather(*tasks, return_exceptions=False))

        # ----------------------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------------------
        n_errors = sum(1 for r in results if r.error is not None)
        valid_results = [r for r in results if r.error is None]

        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        lat_stats = self._compute_latency_percentiles(latencies)

        mean_sim = (
            sum(r.semantic_similarity for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )
        mean_f1 = (
            sum(r.tool_f1 for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )
        mean_safety = (
            sum(r.safety_citation_score for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )
        mean_overall = (
            sum(r.overall_score for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )

        # Per-category breakdown
        categories: Dict[str, List[float]] = {}
        for case, result in zip(eval_cases, results):
            if result.error is None:
                cat = case.category
                categories.setdefault(cat, []).append(result.overall_score)

        category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in categories.items()
        }

        suite_result = EvalSuiteResult(
            suite_name=suite_name,
            n_cases=len(eval_cases),
            n_errors=n_errors,
            run_timestamp=run_timestamp,
            mean_semantic_similarity=mean_sim,
            mean_tool_f1=mean_f1,
            mean_safety_citation_score=mean_safety,
            mean_overall_score=mean_overall,
            mean_latency_ms=lat_stats["mean"],
            p50_latency_ms=lat_stats["p50"],
            p95_latency_ms=lat_stats["p95"],
            p99_latency_ms=lat_stats["p99"],
            min_latency_ms=lat_stats["min"],
            max_latency_ms=lat_stats["max"],
            category_scores=category_scores,
            results=results,
        )

        logger.info("Evaluation suite complete:\n%s", suite_result.summary())
        return suite_result

    def save_results(
        self, suite_result: EvalSuiteResult, output_path: str
    ) -> None:
        """Save evaluation results to a JSON file."""
        data: Dict[str, Any] = suite_result.to_dict()
        data["individual_results"] = [
            {
                "case_id": r.case_id,
                "query": r.query,
                "latency_ms": r.latency_ms,
                "semantic_similarity": r.semantic_similarity,
                "tool_precision": r.tool_precision,
                "tool_recall": r.tool_recall,
                "tool_f1": r.tool_f1,
                "safety_citation_score": r.safety_citation_score,
                "overall_score": r.overall_score,
                "tools_used": r.actual_tools_used,
                "error": r.error,
                "timestamp": r.timestamp,
            }
            for r in suite_result.results
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Evaluation results saved to %s", output_path)

    def compare_runs(
        self,
        baseline: EvalSuiteResult,
        candidate: EvalSuiteResult,
    ) -> Dict[str, Any]:
        """
        Compare two evaluation suite runs and return delta metrics.

        Positive delta means candidate improved over baseline.
        """
        def delta(a: float, b: float) -> float:
            return round(b - a, 4)

        return {
            "baseline_run": baseline.run_timestamp,
            "candidate_run": candidate.run_timestamp,
            "accuracy_delta": {
                "semantic_similarity": delta(
                    baseline.mean_semantic_similarity,
                    candidate.mean_semantic_similarity,
                ),
                "tool_f1": delta(baseline.mean_tool_f1, candidate.mean_tool_f1),
                "safety_citation": delta(
                    baseline.mean_safety_citation_score,
                    candidate.mean_safety_citation_score,
                ),
                "overall": delta(baseline.mean_overall_score, candidate.mean_overall_score),
            },
            "latency_delta_ms": {
                "p50": delta(baseline.p50_latency_ms, candidate.p50_latency_ms),
                "p95": delta(baseline.p95_latency_ms, candidate.p95_latency_ms),
                "p99": delta(baseline.p99_latency_ms, candidate.p99_latency_ms),
            },
            "regression": candidate.mean_overall_score < baseline.mean_overall_score - 0.05,
            "improvement": candidate.mean_overall_score > baseline.mean_overall_score + 0.02,
        }
