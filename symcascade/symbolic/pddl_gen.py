"""C1: L2 stage — LLM generates PDDL, Fast Downward solves it.

This is the symbolic chain's no-skeleton path. On success the solved plan is
generalized into a skeleton and stored in the C2 cache, growing the
self-growing scheduler's hit surface.
"""
from __future__ import annotations

from typing import Callable, Protocol

from symcascade.core.types import Query, StageResult
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import extract_skeleton


class PDDLGenLLM(Protocol):
    def generate_pddl(self, query_text: str) -> str: ...


class PDDLGenStage:
    def __init__(
        self,
        llm: PDDLGenLLM,
        fd: FDSolver,
        cache: SkeletonCache,
        domain_pddl: str,
        problem_pddl_fn: Callable[[Query], str],
    ):
        self._llm = llm
        self._fd = fd
        self._cache = cache
        self._domain = domain_pddl
        self._problem_fn = problem_pddl_fn

    def run(self, query: Query) -> StageResult:
        problem_pddl = self._llm.generate_pddl(query.text)
        result: FDSolverResult = self._fd.solve(self._domain, problem_pddl)
        if not result.success:
            return StageResult(answer="", success=False, cost=0.4, latency_ms=0.0)
        # extract skeleton and cache it (C2 self-growth)
        skel = extract_skeleton(result.plan)
        self._cache.put(f"skel_{self._cache.size()}", skel)
        answer = " ".join(s["name"] for s in result.plan)
        return StageResult(answer=answer, success=True, cost=0.4, latency_ms=50.0,
                           confidence=1.0)
