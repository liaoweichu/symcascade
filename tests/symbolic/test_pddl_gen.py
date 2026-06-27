from symcascade.core.types import Query, StageResult
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.pddl_gen import PDDLGenStage
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.skeleton import extract_skeleton


class FakeFD:
    def __init__(self, plan):
        self._plan = plan
        self.calls = 0

    def solve(self, domain_pddl, problem_pddl):
        self.calls += 1
        return FDSolverResult(success=True, plan=self._plan, stderr="", returncode=0)


class FakeLLM:
    def __init__(self, pddl):
        self._pddl = pddl
        self.calls = 0

    def generate_pddl(self, query_text):
        self.calls += 1
        return self._pddl


def test_l2_generates_pddl_then_solves_then_caches_skeleton():
    cache = SkeletonCache(threshold=0.5)
    fd = FakeFD(plan=[{"name": "goto"}, {"name": "take"}, {"name": "goto"},
                      {"name": "put"}])
    llm = FakeLLM(pddl="(goal ...)")
    stage = PDDLGenStage(llm=llm, fd=fd, cache=cache,
                         domain_pddl="d", problem_pddl_fn=lambda q: "p")
    result = stage.run(Query(text="put apple in fridge", embedding=[]))
    assert result.success is True
    assert llm.calls == 1
    assert fd.calls == 1
    assert cache.size() == 1  # skeleton extracted and stored


def test_l2_failure_when_solve_fails():
    cache = SkeletonCache(threshold=0.5)
    fd = FakeFD(plan=[])
    fd.solve = lambda d, p: FDSolverResult(success=False, plan=[], stderr="x", returncode=1)
    llm = FakeLLM(pddl="(goal ...)")
    stage = PDDLGenStage(llm=llm, fd=fd, cache=cache,
                         domain_pddl="d", problem_pddl_fn=lambda q: "p")
    result = stage.run(Query(text="q", embedding=[]))
    assert result.success is False
    assert cache.size() == 0  # nothing cached on failure
