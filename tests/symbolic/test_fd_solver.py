import pytest
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction


def test_solver_uses_injected_runner(monkeypatch):
    """FD is called via subprocess; tests inject a fake runner to avoid the binary."""
    calls = []

    def fake_run(domain_pddl, problem_pddl, timeout, prefix=None):
        calls.append((domain_pddl, problem_pddl, timeout))
        return FDSolverResult(success=True, plan=[{"name": "goto"}, {"name": "take"}],
                              stderr="", returncode=0)

    solver = FDSolver(runner=fake_run, timeout=30)
    result = solver.solve(domain_pddl="(:predicates ...)", problem_pddl="(:goal ...)")
    assert result.success is True
    assert [s["name"] for s in result.plan] == ["goto", "take"]
    assert len(calls) == 1


def test_solver_failure_returns_success_false(monkeypatch):
    def fake_run(domain_pddl, problem_pddl, timeout, prefix=None):
        return FDSolverResult(success=False, plan=[], stderr="unsolvable", returncode=1)
    solver = FDSolver(runner=fake_run)
    result = solver.solve(domain_pddl="d", problem_pddl="p")
    assert result.success is False
    assert result.plan == []


def test_constrained_replan_passes_skeleton_prefix(monkeypatch):
    received = {}

    def fake_run(domain_pddl, problem_pddl, timeout, prefix=None):
        received["prefix"] = prefix
        return FDSolverResult(success=True, plan=[{"name": "goto"}], stderr="", returncode=0)

    solver = FDSolver(runner=fake_run)
    skel = Skeleton(actions=(SkeletonAction(name="goto"), SkeletonAction(name="take")))
    solver.constrained_replan(domain_pddl="d", problem_pddl="p", skeleton=skel)
    assert received["prefix"] == ["goto", "take"]


def test_default_runner_is_subprocess(monkeypatch):
    """Default runner must shell out, not link FD (GPL isolation)."""
    import symcascade.symbolic.fd_solver as mod
    captured = {}

    def fake_subprocess(cmd, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return 0, "Solution found.\n  goto\n  take\n", ""

    monkeypatch.setattr(mod, "_subprocess_run", fake_subprocess)
    monkeypatch.setattr(mod, "_parse_sas_plan", lambda path: [{"name": "goto"}, {"name": "take"}])
    solver = FDSolver(timeout=10)
    result = solver.solve(domain_pddl="d", problem_pddl="p")
    assert captured["timeout"] == 10
    assert "fast-downward" in captured["cmd"][0]
    assert result.success is True
