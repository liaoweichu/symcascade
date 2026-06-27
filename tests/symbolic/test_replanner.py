from symcascade.core.types import Query, StageResult, Tier
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.replanner import SkeletonReplanner
from symcascade.cache.skeleton_cache import SkeletonCache


class FakeFDSolver:
    """Fakes FDSolver so the L1 stage is testable without the binary."""
    def __init__(self, plan_steps):
        self._plan_steps = plan_steps
        self.last_prefix = None

    def constrained_replan(self, domain_pddl, problem_pddl, skeleton):
        self.last_prefix = [a.name for a in skeleton.actions]
        plan = [{"name": n} for n in self._plan_steps]
        return FDSolverResult(success=True, plan=plan, stderr="", returncode=0)


def test_replanner_returns_success_when_solve_succeeds():
    cache = SkeletonCache(threshold=0.5)
    skel = Skeleton(actions=(SkeletonAction(name="goto"), SkeletonAction(name="take")))
    cache.put("k1", skel)
    fd = FakeFDSolver(plan_steps=["goto", "take", "goto", "put"])
    stage = SkeletonReplanner(cache=cache, fd=fd,
                              domain_pddl="d", problem_pddl_fn=lambda q: "p",
                              query_skeleton_fn=lambda q: skel)
    result = stage.run(Query(text="q", embedding=[]))
    assert result.success is True
    assert isinstance(result, StageResult)


def test_replanner_returns_failure_when_no_skeleton_match():
    cache = SkeletonCache(threshold=0.99)  # nothing matches
    fd = FakeFDSolver(plan_steps=[])
    stage = SkeletonReplanner(cache=cache, fd=fd,
                              domain_pddl="d", problem_pddl_fn=lambda q: "p")
    result = stage.run(Query(text="q", embedding=[]))
    assert result.success is False


def test_replanner_passes_skeleton_prefix_to_fd():
    cache = SkeletonCache(threshold=0.5)
    skel = Skeleton(actions=(SkeletonAction(name="goto"), SkeletonAction(name="take")))
    cache.put("k1", skel)
    fd = FakeFDSolver(plan_steps=["goto"])
    stage = SkeletonReplanner(cache=cache, fd=fd,
                              domain_pddl="d", problem_pddl_fn=lambda q: "p")
    # inject a query skeleton that matches k1
    stage._query_skeleton_fn = lambda q: skel
    stage.run(Query(text="q", embedding=[]))
    assert fd.last_prefix == ["goto", "take"]
