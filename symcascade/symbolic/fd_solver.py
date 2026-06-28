"""C1: Fast Downward solver wrapper.

GPL isolation: Fast Downward is GPL-3.0. We invoke it as a subprocess via
`unified-planning` style command-line, never importing its Python. This keeps
SymCascade (Apache-2.0-compatible) free of GPL contamination.

The runner is injectable so tests never need the binary.

Install for real runs: ``pip install fast-downward`` (provides the
``fast-downward`` CLI) or build from https://github.com/aibasel/downward.
The default runner shells out to ``fast-downward <domain> <problem>
--search "lama()"`` and parses the emitted ``sas_plan``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

from symcascade.symbolic.skeleton import Skeleton


@dataclass
class FDSolverResult:
    success: bool
    plan: list[dict] = field(default_factory=list)
    stderr: str = ""
    returncode: int = 0


def _subprocess_run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    """Real subprocess runner. Isolated here so tests can monkeypatch it."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _default_runner(domain_pddl: str, problem_pddl: str, timeout: int,
                    prefix: Optional[list[str]] = None) -> FDSolverResult:
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        dom = os.path.join(d, "d.pddl")
        prob = os.path.join(d, "p.pddl")
        with open(dom, "w") as f:
            f.write(domain_pddl)
        with open(prob, "w") as f:
            f.write(problem_pddl)
        cmd = ["fast-downward", dom, prob, "--search", "lama()"]
        try:
            rc, out, err = _subprocess_run(cmd, timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return FDSolverResult(success=False, stderr="fd unavailable/timeout")
        plan = _parse_sas_plan(os.path.join(d, "sas_plan")) if rc == 0 else []
        return FDSolverResult(success=rc == 0 and bool(plan), plan=plan,
                              stderr=err, returncode=rc)


def _parse_sas_plan(path: str) -> list[dict]:
    import os
    import re
    if not os.path.exists(path):
        return []
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                name = re.sub(r"[()]", "", line).split()[0]
                steps.append({"name": name})
    return steps


class FDSolver:
    def __init__(self, timeout: int = 30,
                 runner: Optional[Callable] = None):
        self._timeout = timeout
        self._runner = runner or _default_runner

    def solve(self, domain_pddl: str, problem_pddl: str) -> FDSolverResult:
        return self._runner(domain_pddl, problem_pddl, self._timeout)

    def constrained_replan(self, domain_pddl: str, problem_pddl: str,
                           skeleton: Skeleton) -> FDSolverResult:
        prefix = [a.name for a in skeleton.actions]
        return self._runner(domain_pddl, problem_pddl, self._timeout, prefix=prefix)
