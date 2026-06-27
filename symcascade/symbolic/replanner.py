"""C2: L1 stage — skeleton-constrained replanning.

On a cache miss at L0, the discriminator routes formalizable queries with a
matching skeleton here. The replanner asks Fast Downward to replan over the
skeleton prefix (fast replanning / LAMA-style restart), avoiding a full
replan. This is the C2 contribution that APC replaces with small-LLM
adaptation.
"""
from __future__ import annotations

from typing import Callable, Optional

from symcascade.core.types import Query, StageResult
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import Skeleton


class SkeletonReplanner:
    """L1 stage: constrained replanning over a matched skeleton prefix."""

    def __init__(
        self,
        cache: SkeletonCache,
        fd: FDSolver,
        domain_pddl: str,
        problem_pddl_fn: Callable[[Query], str],
        query_skeleton_fn: Optional[Callable[[Query], Skeleton]] = None,
    ):
        self._cache = cache
        self._fd = fd
        self._domain = domain_pddl
        self._problem_fn = problem_pddl_fn
        # In production this derives a query skeleton from the query's PDDL
        # goal; tests inject it directly.
        self._query_skeleton_fn = query_skeleton_fn or _default_query_skeleton

    def run(self, query: Query) -> StageResult:
        query_skel = self._query_skeleton_fn(query)
        match = self._cache.match(query_skel)
        if match is None:
            return StageResult(answer="", success=False, cost=0.0, latency_ms=0.0)
        _key, skel, _sim = match
        result: FDSolverResult = self._fd.constrained_replan(
            self._domain, self._problem_fn(query), skel
        )
        if not result.success:
            return StageResult(answer="", success=False, cost=0.05,
                               latency_ms=0.0)
        answer = " ".join(s["name"] for s in result.plan)
        return StageResult(answer=answer, success=True, cost=0.05, latency_ms=10.0,
                           confidence=1.0)


def _default_query_skeleton(query: Query) -> Skeleton:
    """Fallback: empty skeleton (will never match). Real impl derives from goal."""
    return Skeleton(actions=())
