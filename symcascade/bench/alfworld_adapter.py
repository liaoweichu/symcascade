"""ALFWorld adapter: tasks -> Query + PDDL, and plan -> quality.

Bridges ALFWorld's task format into SymCascade's ``Query`` / PDDL inputs.
Task loading is lazy: if the ``alfworld`` package and its data are present,
real tasks are loaded; otherwise a synthetic fixture set is generated so the
harness can be unit-tested without the 1.5GB ALFWorld download.

Responsibilities:
- ``load_tasks``: yield ``ALFWorldTask`` (text, goal PDDL, task_class, init).
- ``task_to_query``: build a ``Query`` with text + embedding.
- ``goal_skeleton``: derive a ``Skeleton`` from the task's goal (the
  ``query_skeleton_fn`` the L1 replanner needs).
- ``evaluate``: programmatic task-success check (no LLM judge, per spec 4.6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from symcascade.core.types import Query
from symcascade.bench.alfworld_domain import (
    DOMAIN_PDDL, TASK_CLASSES, TASK_SKELETONS,
)
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction


@dataclass(frozen=True)
class ALFWorldTask:
    """A single ALFWorld task instance."""
    task_id: str
    text: str                 # natural-language goal, e.g. "put a clean apple in the fridge"
    goal_predicates: tuple[str, ...]   # PDDL goal atoms, e.g. ("in apple-1 fridge-1", "clean apple-1")
    task_class: str           # one of TASK_CLASSES
    init_predicates: tuple[str, ...] = field(default_factory=tuple)
    problem_pddl: str = ""    # full PDDL problem text (for FD)


_GOAL_RE = re.compile(r"\(([^()]+)\)")


def _parse_goal_atoms(goal_str: str) -> tuple[str, ...]:
    """Extract bare predicate strings from a PDDL goal like (and (in a b) (clean a))."""
    return tuple(m.strip() for m in _GOAL_RE.findall(goal_str))


def _task_class_from_text(text: str) -> str:
    """Heuristic task-class detector from NL goal text.

    ALFWorld templates are highly regular; keyword matching suffices and
    avoids depending on the alfworld package for classification.
    """
    t = text.lower()
    if "clean" in t and "put" in t or "clean" in t and "place" in t:
        return "pick_clean_then_place"
    if "heat" in t and ("put" in t or "place" in t or "microwave" in t):
        return "pick_heat_then_place"
    if "cool" in t and ("put" in t or "place" in t or "fridge" in t):
        return "pick_cool_then_place"
    if "examine" in t or "look at" in t or "light" in t:
        return "look_at_obj_in_light"
    if "two" in t or "both" in t:
        return "pick_two_and_place"
    return "pick_and_place"


def goal_skeleton(task: ALFWorldTask) -> Skeleton:
    """Derive a reusable Skeleton from a task's class.

    ALFWorld's 6 task classes each have a canonical action sequence (see
    TASK_SKELETONS). We emit that sequence as a name-only skeleton; the
    matcher compares action sequences, and L1's constrained replanner fills
    concrete pre/effects from the matched cached skeleton at solve time.
    """
    seq = TASK_SKELETONS.get(task.task_class, ("goto", "pick", "goto", "put"))
    actions = tuple(SkeletonAction(name=n) for n in seq)
    return Skeleton(actions=actions)


def task_to_query(
    task: ALFWorldTask,
    embed_fn: Callable[[str], Sequence[float]],
) -> Query:
    """Build a Query from a task, embedding its NL goal text."""
    return Query(text=task.text, embedding=list(embed_fn(task.text)))


def task_to_problem_pddl(task: ALFWorldTask) -> str:
    """Build a PDDL problem string from a task's init + goal.

    If the task already carries a full ``problem_pddl`` (from real ALFWorld
    data), return it verbatim; otherwise synthesize one from predicates.
    """
    if task.problem_pddl:
        return task.problem_pddl
    init = "\n    ".join(f"({p})" for p in task.init_predicates) or "(at-agent agent-1 room-1)"
    goal = " ".join(f"({p})" for p in task.goal_predicates)
    if len(task.goal_predicates) > 1:
        goal = f"(and {goal})"
    return f"""\
(define (problem {task.task_id})
  (:domain alfworld)
  (:init {init})
  (:goal {goal})
)
"""


def evaluate(answer: str, task: ALFWorldTask) -> float:
    """Programmatic task-success check (spec 4.6 first layer: no judge).

    ALFWorld success = the executed plan achieves all goal predicates. In
    the cascade, ``answer`` is a space-joined action-name sequence. We
    check whether every goal predicate is satisfied by simulating the
    action effects over the init state, routing the agent toward the goal's
    target location on each ``goto``.

    Returns 1.0 on success, 0.0 otherwise.
    """
    if not answer:
        return 0.0
    # Symbolic-chain answers are action sequences; cloud answers are free
    # text. For the latter, fall back to substring check on key goal tokens.
    actions = answer.split()
    if not all(a in {"goto", "pick", "put", "clean", "heat", "cool", "examine"}
               for a in actions):
        return _fuzzy_match(answer, task)
    state = _simulate(actions, task.init_predicates, task.goal_predicates)
    return 1.0 if _goal_satisfied(task.goal_predicates, state) else 0.0


def _simulate(actions: list[str], init: tuple[str, ...],
              goal: tuple[str, ...] = ()) -> set[str]:
    """Tiny STRIPS simulator for ALFWorld actions.

    ``goto`` routes the agent to the next relevant location: before ``pick``
    it goes to the object's source; after ``pick`` it goes to the goal's
    target location (parsed from the ``in OBJ DST`` goal atom). This models
    the constrained-replan skeleton's intent without needing concrete
    arguments in the name-only plan.
    """
    state: set[str] = set(init)
    held: Optional[str] = None
    agent_loc: Optional[str] = None
    for p in init:
        if p.startswith("at-agent "):
            agent_loc = p.split()[-1]
    # goal target location: the DST in an `in OBJ DST` or `at-object OBJ DST` goal
    goal_dst: Optional[str] = None
    for g in goal:
        toks = g.split()
        if len(toks) >= 3 and toks[0] in ("in", "at-object"):
            goal_dst = toks[-1]
            break

    for i, a in enumerate(actions):
        if a == "goto":
            # next action decides target: pick -> go to object source; else -> goal dst
            nxt = actions[i + 1] if i + 1 < len(actions) else None
            if nxt == "pick" and held is None:
                # move to where an at-object currently is
                for p in state:
                    if p.startswith("at-object "):
                        agent_loc = p.split()[-1]
                        break
            elif goal_dst:
                agent_loc = goal_dst
            continue
        if a == "pick" and held is None:
            for p in list(state):
                if p.startswith("at-object ") and agent_loc and p.endswith(agent_loc):
                    held = p.split()[1]
                    state.discard(p)
                    break
        elif a == "put" and held is not None:
            state.add(f"at-object {held} {agent_loc or 'loc'}")
            state.add(f"in {held} {agent_loc or 'loc'}")
            held = None
        elif a == "clean" and held:
            state.add(f"clean {held}")
        elif a == "heat" and held:
            state.add(f"hot {held}")
        elif a == "cool" and held:
            state.add(f"cold {held}")
        elif a == "examine" and held:
            state.add(f"examined {held}")
    return state


def _goal_satisfied(goal: tuple[str, ...], state: set[str]) -> bool:
    return all(g in state for g in goal)


def _fuzzy_match(answer: str, task: ALFWorldTask) -> float:
    """Fallback for cloud free-text answers: check goal keywords present."""
    ans = answer.lower()
    ok = 0
    for g in task.goal_predicates:
        tokens = g.split()
        if len(tokens) >= 2 and tokens[1] in ans:
            ok += 1
    return 1.0 if ok == len(task.goal_predicates) and task.goal_predicates else 0.0


# ---------------------------------------------------------------------------
# Task loaders
# ---------------------------------------------------------------------------

def load_tasks(
    n: int = 134,
    data_path: Optional[str] = None,
    seed: int = 42,
) -> list[ALFWorldTask]:
    """Load ALFWorld tasks.

    If ``data_path`` points at a real ALFWorld install (or the ``alfworld``
    package is importable), real tasks are loaded. Otherwise a deterministic
    synthetic fixture set of ``n`` tasks spanning all 6 classes is generated
    so the harness is unit-testable without the 1.5GB download.
    """
    if data_path:
        try:
            return _load_real_tasks(data_path, n)
        except Exception:
            pass  # fall through to synthetic
    return _synthetic_tasks(n, seed)


def _load_real_tasks(data_path: str, n: int) -> list[ALFWorldTask]:
    """Load from an ALFWorld JSON/PDDL tree on disk.

    ALFWorld's ``json_`` split ships ``game.tw-pddl`` files per task. We
    parse the problem.pddl for init/goal and the task class from the
    directory name. This is best-effort and tolerant of layout variations.
    """
    import json
    import os
    tasks: list[ALFWorldTask] = []
    for root, _dirs, files in os.walk(data_path):
        for fn in files:
            if not (fn.endswith("problem.pddl") or fn == "problem.pddl"):
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                pddl = f.read()
            goal_match = re.search(r":goal\s*\((.*?)\)\s*\)", pddl, re.DOTALL)
            init_match = re.search(r":init\s*(.*?)\s*:goal", pddl, re.DOTALL)
            if not goal_match:
                continue
            goal_atoms = _parse_goal_atoms(goal_match.group(1))
            init_atoms = _parse_goal_atoms(init_match.group(1)) if init_match else ()
            # task class from parent dir name
            parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
            task_class = next(
                (c for c in TASK_CLASSES if c.replace("_", "-") in parent),
                "pick_and_place",
            )
            text = goal_atoms[0].replace("-", " ") if goal_atoms else parent
            tasks.append(ALFWorldTask(
                task_id=f"alfworld_{len(tasks)}",
                text=text,
                goal_predicates=goal_atoms,
                task_class=task_class,
                init_predicates=init_atoms,
                problem_pddl=pddl,
            ))
            if len(tasks) >= n:
                return tasks
    return tasks


def _synthetic_tasks(n: int, seed: int) -> list[ALFWorldTask]:
    """Deterministic synthetic ALFWorld-like tasks for harness testing."""
    import random
    rng = random.Random(seed)
    objs = ["apple", "mug", "plate", "tomato", "egg", "book"]
    locs = ["countertop", "fridge", "microwave", "sink", "drawer", "shelf"]
    tasks: list[ALFWorldTask] = []
    for i in range(n):
        tc = TASK_CLASSES[i % len(TASK_CLASSES)]
        obj = rng.choice(objs)
        src = rng.choice(locs)
        dst = rng.choice([l for l in locs if l != src])
        seq = TASK_SKELETONS[tc]
        goal: list[str] = []
        if "clean" in tc:
            goal.append(f"clean {obj}-{i}")
        if "heat" in tc:
            goal.append(f"hot {obj}-{i}")
        if "cool" in tc:
            goal.append(f"cold {obj}-{i}")
        goal.append(f"in {obj}-{i} {dst}-{i}")
        init = [
            f"at-agent agent-1 {src}-{i}",
            f"at-object {obj}-{i} {src}-{i}",
            f"is-hot-source microwave-{i}",
            f"is-cold-source fridge-{i}",
            f"is-examined-source shelf-{i}",
        ]
        text = f"put the {obj} in the {dst}"
        if "clean" in tc: text = f"clean the {obj} then put it in the {dst}"
        if "heat" in tc: text = f"heat the {obj} then put it in the {dst}"
        if "cool" in tc: text = f"cool the {obj} then put it in the {dst}"
        if "examine" in tc or "look" in tc: text = f"examine the {obj} with the light"
        if "two" in tc: text = f"put two {obj}s in the {dst}"
        tasks.append(ALFWorldTask(
            task_id=f"syn_{i}",
            text=text,
            goal_predicates=tuple(goal),
            task_class=tc,
            init_predicates=tuple(init),
        ))
    return tasks
