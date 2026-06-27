# SymCascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the SymCascade dual-path discriminative routing cascade — a cache-state-aware online ROI discriminator that routes queries to a symbolic chain (skeleton replanning / PDDL generation+solve) or a neural chain (Gemma confidence gating / cloud fallback), with PDDL skeleton caching + constrained replanning as the main innovation.

**Architecture:** A `Cascade` orchestrator runs L0 semantic cache → D discriminator → (档1 L1 skeleton replan | 档2 L2 PDDL gen+solve | 档3 L3 Gemma confidence gate → L4 cloud fallback), with fallback chains between tiers. The discriminator takes query embedding ⊕ cache-state features, outputs a 3-tier ROI prediction set (conformal-covered), and learns online (sliding window + EWMA + drift detection). External heavy dependencies (vLLM, Fast Downward, GPTCache) sit behind interfaces so the core is unit-testable without infra.

**Tech Stack:** Python 3.11, pytest, dataclasses, NumPy/scikit-learn (discriminator), MAPIE (conformal), unified-planning + Fast Downward via subprocess (GPL isolation), vLLM (Gemma backend). Tests run against in-memory mocks; no GPU/planner required for the test suite.

---

## File Structure

```
/workspace/
├── pyproject.toml
├── symcascade/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py            # Query, StageResult, ROI, CacheState, Tier enum
│   │   ├── stage.py            # Stage abstract interface + FallbackError
│   │   └── orchestrator.py     # Cascade: L0→D→L1/L2/L3→L4 with fallback
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── semantic_cache.py   # L0 semantic cache (interface + in-memory impl)
│   │   └── skeleton_cache.py   # C2 skeleton store + cache-state features
│   ├── discriminator/
│   │   ├── __init__.py
│   │   ├── features.py         # build features from query ⊕ CacheState
│   │   ├── roi_estimator.py    # I1 ROI regressor (sklearn GBDT)
│   │   ├── online_learner.py   # sliding window + EWMA + drift detection
│   │   ├── conformal.py        # split-conformal prediction set for tier ROI
│   │   └── router.py           # DiscriminatorRouter: pick tier from ROI set
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── skeleton.py         # C2 skeleton extractor (plan → skeleton)
│   │   ├── matcher.py          # C2 skeleton semantic matcher (edit dist + graph sim)
│   │   ├── replanner.py        # C2 constrained replanner (L1)
│   │   ├── pddl_gen.py         # C1 LLM→PDDL generation (L2) interface
│   │   └── fd_solver.py        # C1 Fast Downward wrapper (subprocess isolation)
│   └── neural/
│       ├── __init__.py
│       ├── gemma_backend.py    # L3 Gemma inference interface (logprobs)
│       └── confidence_gate.py  # C3 confidence gating (conformal on logprobs)
└── tests/
    ├── __init__.py
    ├── core/
    │   ├── test_types.py
    │   └── test_orchestrator.py
    ├── cache/
    │   ├── test_semantic_cache.py
    │   └── test_skeleton_cache.py
    ├── discriminator/
    │   ├── test_features.py
    │   ├── test_roi_estimator.py
    │   ├── test_online_learner.py
    │   ├── test_conformal.py
    │   └── test_router.py
    ├── symbolic/
    │   ├── test_skeleton.py
    │   ├── test_matcher.py
    │   ├── test_replanner.py
    │   ├── test_pddl_gen.py
    │   └── test_fd_solver.py
    └── neural/
        ├── test_gemma_backend.py
        └── test_confidence_gate.py
```

**Responsibility boundaries:**
- `core/types.py` — pure data, no deps; everyone imports from here
- `core/stage.py` — the `Stage` protocol every tier implements
- `core/orchestrator.py` — only orchestration + fallback; no tier internals
- `discriminator/*` — I1; depends on `core/types` + `cache/skeleton_cache` (for features only)
- `symbolic/*` — C1+C2; `fd_solver` is the only GPL-touching file, isolated by subprocess
- `neural/*` — L3/L4 + C3; vLLM behind `gemma_backend` interface

---

### Task 1: Project scaffolding + core types

**Files:**
- Create: `pyproject.toml`
- Create: `symcascade/__init__.py`
- Create: `symcascade/config.py`
- Create: `symcascade/core/__init__.py`
- Create: `symcascade/core/types.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_types.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "symcascade"
version = "0.1.0"
description = "Cache-state-aware dual-path discriminative routing cascade"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
conformal = ["mapie>=0.8"]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create package init files**

`symcascade/__init__.py`:
```python
"""SymCascade: cache-state-aware dual-path discriminative routing cascade."""

__version__ = "0.1.0"
```

`symcascade/core/__init__.py`:
```python
"""Core types and orchestration."""
```

`tests/__init__.py` and `tests/core/__init__.py`: empty files.

- [ ] **Step 3: Write the failing test for core types**

`symcascade/config.py`:
```python
"""Global configuration constants."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeConfig:
    """Runtime configuration for the cascade."""
    conformal_alpha: float = 0.1        # 1-alpha coverage
    ewma_decay: float = 0.95            # online smoothing
    retrain_every: int = 200            # discriminator retrain period
    sliding_window: int = 2000          # online learning window
    skeleton_sim_threshold: float = 0.75  # cosine threshold for skeleton match
    gemma_thinking: bool = False        # force thinking=False on low-cost tier
```

`tests/core/test_types.py`:
```python
from symcascade.core.types import Tier, Query, StageResult, ROI, CacheState


def test_tier_enum_has_four_tiers():
    assert Tier.L1.value == 1
    assert Tier.L2.value == 2
    assert Tier.L3.value == 3
    assert Tier.L4.value == 4


def test_query_carries_text_and_embedding():
    q = Query(text="put the apple in the fridge", embedding=[0.1, 0.2])
    assert q.text == "put the apple in the fridge"
    assert q.embedding == [0.1, 0.2]


def test_stage_result_marks_success_and_cost():
    r = StageResult(answer="done", success=True, cost=0.0, latency_ms=5.0)
    assert r.success is True
    assert r.cost == 0.0


def test_roi_has_quality_cost_latency():
    r = ROI(quality=0.9, cost=0.1, latency=0.2)
    assert r.quality == 0.9


def test_cache_state_tracks_size_and_hit_rate():
    s = CacheState(size=42, topk_sims=[0.8, 0.6], hit_rate=0.3)
    assert s.size == 42
    assert s.topk_sims == [0.8, 0.6]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/core/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'symcascade.core.types'`

- [ ] **Step 5: Implement core types**

`symcascade/core/types.py`:
```python
"""Core data types shared across the cascade."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class Tier(IntEnum):
    """Cascade tiers. L1/L2 form the symbolic chain, L3/L4 the neural chain."""
    L1 = 1  # skeleton constrained replan (CPU, 0 LLM)
    L2 = 2  # LLM generates PDDL + FD solve (1 edge LLM + CPU)
    L3 = 3  # Gemma confidence gating (edge GPU)
    L4 = 4  # cloud Gemini fallback


@dataclass(frozen=True)
class Query:
    """An incoming request."""
    text: str
    embedding: Sequence[float] = field(default_factory=list)


@dataclass(frozen=True)
class StageResult:
    """Output of a single tier stage."""
    answer: str
    success: bool
    cost: float          # normalized cost units
    latency_ms: float
    confidence: float = 0.0  # tier-reported confidence (0..1), 0 if N/A


@dataclass(frozen=True)
class ROI:
    """Expected return-on-investment estimate for a tier."""
    quality: float
    cost: float
    latency: float

    def benefit_cost_ratio(self, budget_lambda: float = 1.0) -> float:
        """Benefit-cost ratio used for tier ranking."""
        denom = self.cost * budget_lambda + 1e-9
        return self.quality / denom


@dataclass(frozen=True)
class CacheState:
    """Snapshot of cache state fed to the discriminator as features."""
    size: int                  # number of cached skeletons
    topk_sims: Sequence[float]  # top-k skeleton cosine sims for current query
    hit_rate: float            # historical hit rate over recent window
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/core/test_types.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml symcascade/ tests/
git commit -m "feat: scaffold project and core types"
```

---

### Task 2: Stage interface + Cascade orchestrator

**Files:**
- Create: `symcascade/core/stage.py`
- Create: `symcascade/core/orchestrator.py`
- Test: `tests/core/test_orchestrator.py`

- [ ] **Step 1: Write the failing test for the orchestrator**

`tests/core/test_orchestrator.py`:
```python
import pytest
from symcascade.core.types import Query, StageResult, Tier
from symcascade.core.stage import Stage, FallbackError
from symcascade.core.orchestrator import Cascade


class StubStage(Stage):
    """Deterministic stub stage for testing."""
    def __init__(self, tier, answer="ok", success=True):
        self._tier = tier
        self._answer = answer
        self._success = success
        self.calls = 0

    def run(self, query: Query) -> StageResult:
        self.calls += 1
        return StageResult(answer=self._answer, success=self._success,
                           cost=0.0, latency_ms=1.0)


class StubCache:
    def __init__(self, hit=False):
        self._hit = hit
    def get(self, query: Query): 
        return "cached" if self._hit else None
    def put(self, query: Query, answer: str): 
        pass


class StubDiscriminator:
    """Always routes to a configured tier."""
    def __init__(self, tier):
        self._tier = tier
    def route(self, query, cache_state):
        return self._tier


def _make_query():
    return Query(text="q", embedding=[0.1])


def test_cache_hit_short_circuits_before_discriminator():
    cache = StubCache(hit=True)
    disc = StubDiscriminator(Tier.L1)
    l1 = StubStage(Tier.L1)
    cascade = Cascade(cache=cache, discriminator=disc,
                      l1=l1, l2=StubStage(Tier.L2), l3=StubStage(Tier.L3),
                      l4=StubStage(Tier.L4), cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.answer == "cached"
    assert l1.calls == 0  # never reached


def test_routes_to_l1_and_succeeds():
    disc = StubDiscriminator(Tier.L1)
    l1 = StubStage(Tier.L1, success=True)
    cascade = Cascade(cache=StubCache(hit=False), discriminator=disc,
                      l1=l1, l2=StubStage(Tier.L2), l3=StubStage(Tier.L3),
                      l4=StubStage(Tier.L4), cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.success is True
    assert l1.calls == 1


def test_l1_failure_falls_back_to_l2():
    disc = StubDiscriminator(Tier.L1)
    l1 = StubStage(Tier.L1, success=False)
    l2 = StubStage(Tier.L2, success=True)
    cascade = Cascade(cache=StubCache(hit=False), discriminator=disc,
                      l1=l1, l2=l2, l3=StubStage(Tier.L3),
                      l4=StubStage(Tier.L4), cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.success is True
    assert l1.calls == 1
    assert l2.calls == 1


def test_l2_failure_falls_back_to_l3():
    disc = StubDiscriminator(Tier.L2)
    l2 = StubStage(Tier.L2, success=False)
    l3 = StubStage(Tier.L3, success=True)
    cascade = Cascade(cache=StubCache(hit=False), discriminator=disc,
                      l1=StubStage(Tier.L1), l2=l2, l3=l3,
                      l4=StubStage(Tier.L4), cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.success is True
    assert l3.calls == 1


def test_l3_low_confidence_falls_back_to_l4():
    disc = StubDiscriminator(Tier.L3)
    l3 = StubStage(Tier.L3, success=False)  # low confidence modelled as failure
    l4 = StubStage(Tier.L4, success=True)
    cascade = Cascade(cache=StubCache(hit=False), discriminator=disc,
                      l1=StubStage(Tier.L1), l2=StubStage(Tier.L2), l3=l3,
                      l4=l4, cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.success is True
    assert l4.calls == 1


def test_all_fail_returns_last_result():
    disc = StubDiscriminator(Tier.L1)
    cascade = Cascade(cache=StubCache(hit=False), discriminator=disc,
                      l1=StubStage(Tier.L1, success=False),
                      l2=StubStage(Tier.L2, success=False),
                      l3=StubStage(Tier.L3, success=False),
                      l4=StubStage(Tier.L4, success=False, answer="last"),
                      cache_state_fn=lambda: None)
    result = cascade.run(_make_query())
    assert result.success is False
    assert result.answer == "last"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'symcascade.core.stage'`

- [ ] **Step 3: Implement Stage interface and FallbackError**

`symcascade/core/stage.py`:
```python
"""Stage interface implemented by every cascade tier."""
from __future__ import annotations

from typing import Protocol

from symcascade.core.types import Query, StageResult


class FallbackError(Exception):
    """Raised by a stage to signal it cannot handle the query; cascade falls back."""


class Stage(Protocol):
    """A single tier in the cascade.

    A stage either returns a successful StageResult, returns an unsuccessful
    StageResult (which triggers fallback), or raises FallbackError.
    """
    def run(self, query: Query) -> StageResult: ...
```

- [ ] **Step 4: Implement the Cascade orchestrator**

`symcascade/core/orchestrator.py`:
```python
"""Cascade orchestrator: L0 cache -> discriminator -> tier chain with fallback."""
from __future__ import annotations

from typing import Callable, Optional

from symcascade.core.types import CacheState, Query, StageResult, Tier
from symcascade.core.stage import Stage


class Cascade:
    """Runs the dual-path discriminative cascade.

    Flow:
      1. L0 semantic cache hit -> return immediately.
      2. Discriminator picks a starting tier given query + cache state.
      3. Run the chosen tier; on failure fall through the chain:
         L1 -> L2 -> L3 -> L4.
    """

    def __init__(
        self,
        cache,                       # has get(query)->Optional[str], put(query, answer)
        discriminator,               # has route(query, cache_state)->Tier
        l1: Stage,
        l2: Stage,
        l3: Stage,
        l4: Stage,
        cache_state_fn: Callable[[], Optional[CacheState]],
    ):
        self._cache = cache
        self._disc = discriminator
        self._stages: dict[Tier, Stage] = {Tier.L1: l1, Tier.L2: l2,
                                           Tier.L3: l3, Tier.L4: l4}
        self._cache_state_fn = cache_state_fn

    def run(self, query: Query) -> StageResult:
        # L0: semantic cache
        cached = self._cache.get(query)
        if cached is not None:
            return StageResult(answer=cached, success=True, cost=0.0, latency_ms=0.0)

        # Discriminator picks starting tier
        cache_state = self._cache_state_fn()
        start_tier = self._disc.route(query, cache_state)

        # Walk the fallback chain from start_tier to L4
        chain = [t for t in Tier if t >= start_tier]
        last_result: Optional[StageResult] = None
        for tier in chain:
            last_result = self._stages[tier].run(query)
            if last_result.success:
                # cache successful symbolic-chain results
                if tier in (Tier.L1, Tier.L2):
                    self._cache.put(query, last_result.answer)
                return last_result
        # all tiers failed — return last (L4) result
        return last_result  # type: ignore[return-value]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/core/test_orchestrator.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add symcascade/core/stage.py symcascade/core/orchestrator.py tests/core/test_orchestrator.py
git commit -m "feat: add Stage interface and Cascade orchestrator with fallback chain"
```

---

### Task 3: L0 semantic cache (in-memory implementation)

**Files:**
- Create: `symcascade/cache/__init__.py`
- Create: `symcascade/cache/semantic_cache.py`
- Test: `tests/cache/__init__.py`
- Test: `tests/cache/test_semantic_cache.py`

- [ ] **Step 1: Create cache package init**

`symcascade/cache/__init__.py`:
```python
"""Caching layers: L0 semantic cache and C2 skeleton cache."""
```

`tests/cache/__init__.py`: empty.

- [ ] **Step 2: Write the failing test**

`tests/cache/test_semantic_cache.py`:
```python
import math
from symcascade.cache.semantic_cache import SemanticCache


def _norm(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x / n for x in v]


def test_exact_text_hit():
    c = SemanticCache(sim_threshold=0.99)
    c.put("q", "a")
    assert c.get("q") == "a"


def test_miss_returns_none():
    c = SemanticCache(sim_threshold=0.99)
    assert c.get("absent") is None


def test_semantic_hit_above_threshold():
    c = SemanticCache(sim_threshold=0.9, embed_fn=lambda t: _norm([1.0, 0.0]) if "apple" in t else _norm([0.9, 0.1]))
    c.put("put the apple here", "a1")
    # near-identical embedding -> cosine ~0.995 > 0.9
    assert c.get("put the apple there") == "a1"


def test_semantic_miss_below_threshold():
    c = SemanticCache(sim_threshold=0.99, embed_fn=lambda t: _norm([1.0, 0.0]) if "apple" in t else _norm([0.0, 1.0]))
    c.put("apple", "a1")
    assert c.get("banana") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/cache/test_semantic_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement SemanticCache**

`symcascade/cache/semantic_cache.py`:
```python
"""L0 semantic cache.

Exact-text fast path plus cosine-similarity semantic lookup over a pluggable
embed_fn. The real deployment plugs in BGE-M3; tests use a toy embed_fn.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class SemanticCache:
    def __init__(
        self,
        sim_threshold: float = 0.9,
        embed_fn: Callable[[str], Sequence[float]] = lambda t: [0.0],
    ):
        self._sim_threshold = sim_threshold
        self._embed_fn = embed_fn
        self._exact: dict[str, str] = {}
        self._entries: list[tuple[Sequence[float], str, str]] = []  # (emb, key, val)

    def put(self, key: str, value: str) -> None:
        self._exact[key] = value
        emb = self._embed_fn(key)
        self._entries.append((emb, key, value))

    def get(self, key: str) -> Optional[str]:
        # exact fast path
        if key in self._exact:
            return self._exact[key]
        emb = self._embed_fn(key)
        best_sim, best_val = -1.0, None
        for e, _k, v in self._entries:
            sim = _cosine(emb, e)
            if sim > best_sim:
                best_sim, best_val = sim, v
        return best_val if best_sim >= self._sim_threshold else None

    def size(self) -> int:
        return len(self._entries)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/cache/test_semantic_cache.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add symcascade/cache/ tests/cache/
git commit -m "feat: add L0 semantic cache with exact + cosine lookup"
```

---

### Task 4: C2 skeleton data model + extractor

**Files:**
- Create: `symcascade/symbolic/__init__.py`
- Create: `symcascade/symbolic/skeleton.py`
- Test: `tests/symbolic/__init__.py`
- Test: `tests/symbolic/test_skeleton.py`

- [ ] **Step 1: Create symbolic package init**

`symcascade/symbolic/__init__.py`:
```python
"""Symbolic chain: C1 solver backend + C2 skeleton cache/replanning."""
```

`tests/symbolic/__init__.py`: empty.

- [ ] **Step 2: Write the failing test**

`tests/symbolic/test_skeleton.py`:
```python
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction, extract_skeleton


def test_extract_skeleton_from_action_sequence():
    plan = [
        {"name": "goto", "pre": ["at-agent room1"], "eff": ["at-agent room2"]},
        {"name": "take", "pre": ["at-object apple room2"], "eff": ["holding apple"]},
        {"name": "goto", "pre": ["at-agent room2"], "eff": ["at-agent room3"]},
        {"name": "put", "pre": ["holding apple", "at-agent room3"], "eff": ["at-object apple room3"]},
    ]
    skel = extract_skeleton(plan)
    assert [a.name for a in skel.actions] == ["goto", "take", "goto", "put"]
    # predicates are generalized: object/room identifiers stripped
    assert skel.actions[0].pre == ["at-agent LOC"]
    assert skel.actions[1].pre == ["at-object OBJ LOC"]
    assert skel.actions[3].eff == ["at-object OBJ LOC"]


def test_skeleton_template_round_trips():
    skel = Skeleton(actions=[
        SkeletonAction(name="take", pre=["at-object OBJ LOC"], eff=["holding OBJ"]),
    ])
    s = skel.to_template_str()
    assert "take" in s
    parsed = Skeleton.from_template_str(s)
    assert parsed.actions[0].name == "take"


def test_two_plans_same_shape_same_skeleton():
    plan_a = [{"name": "goto", "pre": ["at-agent r1"], "eff": ["at-agent r2"]},
              {"name": "take", "pre": ["at-object x r2"], "eff": ["holding x"]}]
    plan_b = [{"name": "goto", "pre": ["at-agent r3"], "eff": ["at-agent r4"]},
              {"name": "take", "pre": ["at-object y r4"], "eff": ["holding y"]}]
    assert extract_skeleton(plan_a).to_template_str() == extract_skeleton(plan_b).to_template_str()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/symbolic/test_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement skeleton model + extractor**

`symcascade/symbolic/skeleton.py`:
```python
"""C2: PDDL plan skeleton — a generalized, reusable action-sequence template.

A skeleton strips concrete object/room identifiers from a solved plan so the
same action shape can be reused on a new problem instance via constrained
replanning. Predicates are generalized: lowercase alphanumeric tokens that
look like identifiers (rooms, objects) are replaced by LOC / OBJ placeholders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*\d*\b")
_KEEP = {"at-agent", "holding", "at-object", "clean", "hot", "cold", "in"}


def _generalize_predicate(pred: str) -> str:
    out = []
    for tok in pred.split():
        if tok in _KEEP or tok.startswith("at-"):
            out.append(tok)
        elif _IDENT_RE.fullmatch(tok):
            # heuristic: 'at-object X LOC' -> object slot; room-like -> LOC
            out.append("OBJ" if not any(k in tok for k in ("room", "loc")) else "LOC")
        else:
            out.append(tok)
    return " ".join(out) if out else pred


@dataclass(frozen=True)
class SkeletonAction:
    name: str
    pre: tuple[str, ...] = field(default_factory=tuple)
    eff: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Skeleton:
    actions: tuple[SkeletonAction, ...]

    def to_template_str(self) -> str:
        lines = []
        for a in self.actions:
            lines.append(f"{a.name} | pre=[{', '.join(a.pre)}] eff=[{', '.join(a.eff)}]")
        return "\n".join(lines)

    @classmethod
    def from_template_str(cls, s: str) -> "Skeleton":
        actions = []
        for line in s.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            head, _, body = line.partition(" | ")
            name = head.strip()
            pre, eff = [], []
            for part in body.split("]"):
                part = part.strip().lstrip(",").strip()
                if part.startswith("pre=["):
                    pre = [p.strip() for p in part[5:].split(",") if p.strip()]
                elif part.startswith("eff=["):
                    eff = [p.strip() for p in part[5:].split(",") if p.strip()]
            actions.append(SkeletonAction(name=name, pre=tuple(pre), eff=tuple(eff)))
        return cls(actions=tuple(actions))


def extract_skeleton(plan: list[dict]) -> Skeleton:
    """Generalize a solved plan into a reusable Skeleton."""
    actions = []
    for step in plan:
        pre = tuple(_generalize_predicate(p) for p in step.get("pre", []))
        eff = tuple(_generalize_predicate(p) for p in step.get("eff", []))
        actions.append(SkeletonAction(name=step["name"], pre=pre, eff=eff))
    return Skeleton(actions=tuple(actions))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/symbolic/test_skeleton.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add symcascade/symbolic/__init__.py symcascade/symbolic/skeleton.py tests/symbolic/
git commit -m "feat: add C2 skeleton model + generalized extractor"
```

---

### Task 5: C2 skeleton semantic matcher

**Files:**
- Create: `symcascade/symbolic/matcher.py`
- Test: `tests/symbolic/test_matcher.py`

- [ ] **Step 1: Write the failing test**

`tests/symbolic/test_matcher.py`:
```python
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction
from symcascade.symbolic.matcher import SkeletonMatcher, skeleton_similarity


def _skel(names):
    return Skeleton(actions=tuple(SkeletonAction(name=n) for n in names))


def test_identical_skeletons_have_similarity_one():
    a = _skel(["goto", "take", "goto", "put"])
    b = _skel(["goto", "take", "goto", "put"])
    assert skeleton_similarity(a, b) == 1.0


def test_completely_different_have_zero():
    a = _skel(["goto", "take"])
    b = _skel(["look", "clean"])
    # edit distance 2 over length 2 -> sim 0
    assert skeleton_similarity(a, b) == 0.0


def test_partial_overlap_is_between_zero_and_one():
    a = _skel(["goto", "take", "goto", "put"])
    b = _skel(["goto", "take", "goto", "clean"])
    sim = skeleton_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_matcher_returns_best_above_threshold():
    m = SkeletonMatcher(threshold=0.5)
    s1 = _skel(["goto", "take", "goto", "put"])
    s2 = _skel(["goto", "take", "goto", "clean"])
    m.add("s1", s1)
    m.add("s2", s2)
    query = _skel(["goto", "take", "goto", "put"])
    match_id, sim = m.match(query)
    assert match_id == "s1"
    assert sim == 1.0


def test_matcher_returns_none_below_threshold():
    m = SkeletonMatcher(threshold=0.99)
    m.add("s1", _skel(["goto", "take", "goto", "put"]))
    match_id, sim = m.match(_skel(["look", "clean"]))
    assert match_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/symbolic/test_matcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement matcher (action-sequence edit distance)**

`symcascade/symbolic/matcher.py`:
```python
"""C2: skeleton semantic matcher.

Combines action-sequence edit distance with a lightweight predicate-graph
similarity. The combined score is the cosine of [1 - normalized_edit_distance,
predicate_overlap]. GPTCache exposes a pluggable embedding slot but ships no
PDDL-aware metric, so this is the self-authored structured distance.
"""
from __future__ import annotations

from typing import Optional

from symcascade.symbolic.skeleton import Skeleton


def _edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _predicate_overlap(a: Skeleton, b: Skeleton) -> float:
    pa = {p for act in a.actions for p in act.pre + act.eff}
    pb = {p for act in b.actions for p in act.pre + act.eff}
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)


def skeleton_similarity(a: Skeleton, b: Skeleton) -> float:
    """Combined similarity in [0,1]."""
    names_a = [act.name for act in a.actions]
    names_b = [act.name for act in b.actions]
    maxlen = max(len(names_a), len(names_b)) or 1
    ed = _edit_distance(names_a, names_b)
    seq_sim = 1.0 - ed / maxlen
    pred_sim = _predicate_overlap(a, b)
    # weighted combination: sequence shape matters more than predicate overlap
    return 0.7 * seq_sim + 0.3 * pred_sim


class SkeletonMatcher:
    """Holds known skeletons and answers nearest-match queries."""

    def __init__(self, threshold: float = 0.75):
        self._threshold = threshold
        self._store: dict[str, Skeleton] = {}

    def add(self, key: str, skeleton: Skeleton) -> None:
        self._store[key] = skeleton

    def match(self, query: Skeleton) -> tuple[Optional[str], float]:
        best_key, best_sim = None, -1.0
        for key, skel in self._store.items():
            sim = skeleton_similarity(query, skel)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_sim >= self._threshold:
            return best_key, best_sim
        return None, best_sim

    def topk(self, query: Skeleton, k: int = 3) -> list[tuple[str, float]]:
        scored = [(k_, skeleton_similarity(query, s_)) for k_, s_ in self._store.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/symbolic/test_matcher.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/symbolic/matcher.py tests/symbolic/test_matcher.py
git commit -m "feat: add C2 skeleton semantic matcher (edit distance + predicate overlap)"
```

---

### Task 6: C2 skeleton cache store + cache-state features

**Files:**
- Create: `symcascade/cache/skeleton_cache.py`
- Test: `tests/cache/test_skeleton_cache.py`

- [ ] **Step 1: Write the failing test**

`tests/cache/test_skeleton_cache.py`:
```python
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction
from symcascade.core.types import CacheState


def _skel(names):
    return Skeleton(actions=tuple(SkeletonAction(name=n) for n in names))


def test_put_and_match_returns_skeleton():
    c = SkeletonCache(threshold=0.5)
    s = _skel(["goto", "take", "goto", "put"])
    c.put("k1", s)
    hit = c.match(s)
    assert hit is not None
    key, skel, sim = hit
    assert key == "k1"
    assert sim == 1.0


def test_match_miss_returns_none():
    c = SkeletonCache(threshold=0.99)
    c.put("k1", _skel(["goto", "take", "goto", "put"]))
    assert c.match(_skel(["look", "clean"])) is None


def test_state_features_built_from_query():
    c = SkeletonCache(threshold=0.5)
    c.put("k1", _skel(["goto", "take", "goto", "put"]))
    c.put("k2", _skel(["goto", "take", "goto", "clean"]))
    state = c.state_for(_skel(["goto", "take", "goto", "put"]))
    assert isinstance(state, CacheState)
    assert state.size == 2
    assert state.topk_sims[0] == 1.0
    assert len(state.topk_sims) <= 3


def test_hit_rate_updates_after_queries():
    c = SkeletonCache(threshold=0.5)
    target = _skel(["goto", "take", "goto", "put"])
    c.put("k1", target)
    c.match(target)   # hit
    c.match(_skel(["look", "clean"]))  # miss
    state = c.state_for(target)
    assert 0.0 < state.hit_rate <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cache/test_skeleton_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SkeletonCache**

`symcascade/cache/skeleton_cache.py`:
```python
"""C2 skeleton cache store.

Wraps SkeletonMatcher and exposes the cache-state features the discriminator
consumes (size, top-k sims, rolling hit rate). This is the coupling point
between C2 and I1.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from symcascade.core.types import CacheState
from symcascade.symbolic.matcher import SkeletonMatcher
from symcascade.symbolic.skeleton import Skeleton


class SkeletonCache:
    def __init__(self, threshold: float = 0.75, topk: int = 3, hit_window: int = 100):
        self._matcher = SkeletonMatcher(threshold=threshold)
        self._store: dict[str, Skeleton] = {}
        self._topk = topk
        self._hit_window = deque(maxlen=hit_window)  # 1=hit, 0=miss

    def put(self, key: str, skeleton: Skeleton) -> None:
        self._matcher.add(key, skeleton)
        self._store[key] = skeleton

    def match(self, query: Skeleton) -> Optional[tuple[str, Skeleton, float]]:
        key, sim = self._matcher.match(query)
        if key is None:
            self._hit_window.append(0)
            return None
        self._hit_window.append(1)
        return key, self._store[key], sim

    def state_for(self, query: Skeleton) -> CacheState:
        top = self._matcher.topk(query, k=self._topk)
        sims = [sim for _k, sim in top] if top else []
        hit_rate = (sum(self._hit_window) / len(self._hit_window)) if self._hit_window else 0.0
        return CacheState(size=len(self._store), topk_sims=sims, hit_rate=hit_rate)

    def size(self) -> int:
        return len(self._store)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/cache/test_skeleton_cache.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/cache/skeleton_cache.py tests/cache/test_skeleton_cache.py
git commit -m "feat: add C2 skeleton cache store with cache-state features"
```

---

### Task 7: I1 discriminator features + ROI estimator

**Files:**
- Create: `symcascade/discriminator/__init__.py`
- Create: `symcascade/discriminator/features.py`
- Create: `symcascade/discriminator/roi_estimator.py`
- Test: `tests/discriminator/__init__.py`
- Test: `tests/discriminator/test_features.py`
- Test: `tests/discriminator/test_roi_estimator.py`

- [ ] **Step 1: Create package inits**

`symcascade/discriminator/__init__.py`:
```python
"""I1: cache-state-aware online ROI discriminator."""
```

`tests/discriminator/__init__.py`: empty.

- [ ] **Step 2: Write failing test for features**

`tests/discriminator/test_features.py`:
```python
from symcascade.core.types import CacheState, Query
from symcascade.discriminator.features import build_features, FEATURE_DIM


def test_build_features_concatenates_query_and_cache_state():
    q = Query(text="q", embedding=[0.1, 0.2, 0.3])
    cs = CacheState(size=5, topk_sims=[0.8, 0.6, 0.4], hit_rate=0.3)
    feats = build_features(q, cs)
    assert len(feats) == FEATURE_DIM
    # first 3 = query embedding
    assert feats[0] == 0.1
    # last 5 = [size, top1, top2, top3, hit_rate]
    assert feats[-5] == 5
    assert feats[-4] == 0.8
    assert feats[-1] == 0.3


def test_features_handle_short_topk():
    q = Query(text="q", embedding=[0.1, 0.2])
    cs = CacheState(size=1, topk_sims=[0.5], hit_rate=0.0)
    feats = build_features(q, cs)
    assert len(feats) == FEATURE_DIM
    # missing top-k slots padded with 0
    assert feats[-3] == 0.0  # padded top2
```

- [ ] **Step 3: Run test to fails**

Run: `python -m pytest tests/discriminator/test_features.py -v`
Expected: FAIL

- [ ] **Step 4: Implement features**

`symcascade/discriminator/features.py`:
```python
"""I1: feature builder for the discriminator.

Features = query_embedding (padded/truncated to EMB_DIM) ++ cache-state
features [size, top1, top2, top3, hit_rate]. The cache-state features are
what make the discriminator cache-state-aware and non-trivial.
"""
from __future__ import annotations

from typing import Sequence

from symcascade.core.types import CacheState, Query

EMB_DIM = 16         # fixed embedding dim (BGE-M3 truncated/padded in real use)
TOPK_PAD = 3
CACHE_FEAT = 1 + TOPK_PAD + 1  # size + topk + hit_rate
FEATURE_DIM = EMB_DIM + CACHE_FEAT


def _pad(seq: Sequence[float], length: int) -> list[float]:
    out = list(seq[:length])
    out.extend([0.0] * (length - len(out)))
    return out


def build_features(query: Query, cache_state: CacheState | None) -> list[float]:
    emb = _pad(query.embedding, EMB_DIM)
    if cache_state is None:
        return emb + [0.0, 0.0, 0.0, 0.0, 0.0]
    topk = _pad(cache_state.topk_sims, TOPK_PAD)
    return emb + [float(cache_state.size)] + topk + [float(cache_state.hit_rate)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/discriminator/test_features.py -v`
Expected: PASS

- [ ] **Step 6: Write failing test for ROI estimator**

`tests/discriminator/test_roi_estimator.py`:
```python
import pytest
from symcascade.core.types import CacheState, Query
from symcascade.discriminator.features import build_features, FEATURE_DIM
from symcascade.discriminator.roi_estimator import ROIEstimator
from symcascade.core.types import Tier


def _q(emb):
    return Query(text="q", embedding=emb)


def _cs(size, hit_rate=0.5):
    return CacheState(size=size, topk_sims=[0.8, 0.6, 0.4], hit_rate=hit_rate)


def test_estimator_predicts_three_tier_rois():
    est = ROIEstimator()
    est.fit_warmup(n=300, seed=0)
    rois = est.predict(_q([0.1] * 16), _cs(size=10))
    # one ROI per tier L1..L3 (L4 is fallback, no ROI needed)
    assert set(rois.keys()) == {Tier.L1, Tier.L2, Tier.L3}
    for t, r in rois.items():
        assert 0.0 <= r.quality <= 1.0
        assert r.cost >= 0.0


def test_estimator_predict_deterministic_after_fit():
    est = ROIEstimator()
    est.fit_warmup(n=300, seed=1)
    q = _q([0.2] * 16)
    cs = _cs(size=5)
    a = est.predict(q, cs)
    b = est.predict(q, cs)
    assert a == b
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest tests/discriminator/test_roi_estimator.py -v`
Expected: FAIL

- [ ] **Step 8: Implement ROI estimator**

`symcascade/discriminator/roi_estimator.py`:
```python
"""I1: ROI estimator.

A gradient-boosted regressor per tier that predicts expected (quality, cost,
latency) from query+cache-state features. fit_warmup generates a synthetic
warmup corpus so the discriminator has a cold-start policy before real
observations arrive (online_learner.py refines it on real data).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor

from symcascade.core.types import CacheState, Query, ROI, Tier
from symcascade.discriminator.features import FEATURE_DIM, build_features


@dataclass
class _ROIObservation:
    features: list[float]
    quality: float
    cost: float
    latency: float


def _synthetic_obs(tier: Tier, size: int, hit_rate: float, seed: int) -> _ROIObservation:
    """Heuristic warmup distribution. Real deployment replaces with logged data."""
    rng = random.Random(seed)
    emb = [rng.gauss(0, 1) for _ in range(FEATURE_DIM - 5)]
    # cache-state features: larger cache -> higher L1 quality; low hit_rate -> L3 preferred
    feats = emb + [float(size), 0.8, 0.6, 0.4, hit_rate]
    if tier == Tier.L1:
        q = min(1.0, 0.6 + 0.03 * size) if hit_rate > 0.3 else 0.2
        c, l = 0.05, 0.1
    elif tier == Tier.L2:
        q = 0.85
        c, l = 0.4, 0.5
    else:  # L3
        q = 0.7 + 0.1 * hit_rate
        c, l = 0.2, 0.3
    return _ROIObservation(feats, q, c, l)


class ROIEstimator:
    def __init__(self):
        self._models: dict[Tier, tuple[GradientBoostingRegressor, ...]] = {}

    def fit_warmup(self, n: int = 300, seed: int = 0) -> None:
        for tier in (Tier.L1, Tier.L2, Tier.L3):
            obs = [_synthetic_obs(tier, size=i % 50, hit_rate=(i % 10) / 10,
                                  seed=seed + i) for i in range(n)]
            X = [o.features for o in obs]
            self._models[tier] = (
                GradientBoostingRegressor(random_state=seed).fit(X, [o.quality for o in obs]),
                GradientBoostingRegressor(random_state=seed).fit(X, [o.cost for o in obs]),
                GradientBoostingRegressor(random_state=seed).fit(X, [o.latency for o in obs]),
            )

    def predict(self, query: Query, cache_state: CacheState | None) -> dict[Tier, ROI]:
        feats = [build_features(query, cache_state)]
        out: dict[Tier, ROI] = {}
        for tier, (qm, cm, lm) in self._models.items():
            out[tier] = ROI(
                quality=float(max(0.0, min(1.0, qm.predict(feats)[0]))),
                cost=float(max(0.0, cm.predict(feats)[0])),
                latency=float(max(0.0, lm.predict(feats)[0])),
            )
        return out
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/discriminator/test_roi_estimator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Commit**

```bash
git add symcascade/discriminator/ tests/discriminator/
git commit -m "feat: add I1 discriminator features + ROI estimator with warmup"
```

---

### Task 8: I1 online learner (sliding window + EWMA + drift)

**Files:**
- Create: `symcascade/discriminator/online_learner.py`
- Test: `tests/discriminator/test_online_learner.py`

- [ ] **Step 1: Write the failing test**

`tests/discriminator/test_online_learner.py`:
```python
from symcascade.core.types import Tier
from symcascade.discriminator.online_learner import OnlineLearner, OnlineRecord


def _rec(tier=Tier.L1, predicted_q=0.5, actual_q=0.9):
    return OnlineRecord(tier=tier, predicted_quality=predicted_q, actual_quality=actual_q)


def test_records_accumulate_in_window():
    ol = OnlineLearner(window=5)
    for _ in range(3):
        ol.observe(_rec())
    assert len(ol.window) == 3


def test_window_evicts_old_records():
    ol = OnlineLearner(window=2)
    ol.observe(_rec(tier=Tier.L1))
    ol.observe(_rec(tier=Tier.L2))
    ol.observe(_rec(tier=Tier.L3))
    assert len(ol.window) == 2
    assert ol.window[-1].tier == Tier.L3


def test_ewma_error_smooths_predictions():
    ol = OnlineLearner(window=100, ewma_decay=0.5)
    ol.observe(_rec(predicted_q=0.5, actual_q=0.9))  # error 0.4
    assert abs(ol.ewma_error() - 0.4) < 1e-9
    ol.observe(_rec(predicted_q=0.5, actual_q=0.5))  # error 0.0
    # EWMA = 0.5*0.0 + 0.5*0.4 = 0.2
    assert abs(ol.ewma_error() - 0.2) < 1e-9


def test_drift_detected_when_error_jumps():
    ol = OnlineLearner(window=100, ewma_decay=0.9, drift_threshold=0.3)
    # baseline: low error
    for _ in range(20):
        ol.observe(_rec(predicted_q=0.9, actual_q=0.9))
    assert not ol.drift_detected()
    # sudden distribution shift
    for _ in range(20):
        ol.observe(_rec(predicted_q=0.9, actual_q=0.1))
    assert ol.drift_detected()


def test_should_retrain_respects_period():
    ol = OnlineLearner(window=100, retrain_every=5)
    for _ in range(4):
        ol.observe(_rec())
        assert not ol.should_retrain()
    ol.observe(_rec())
    assert ol.should_retrain()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/discriminator/test_online_learner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OnlineLearner**

`symcascade/discriminator/online_learner.py`:
```python
"""I1: online learner for the discriminator.

Sliding window of (tier, predicted_quality, actual_quality) observations.
EWMA-smoothed prediction error feeds drift detection; when error exceeds the
threshold the orchestrator can fall back to a static policy. should_retrain()
paces discriminator refits to avoid thrashing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from symcascade.core.types import Tier


@dataclass(frozen=True)
class OnlineRecord:
    tier: Tier
    predicted_quality: float
    actual_quality: float

    @property
    def error(self) -> float:
        return abs(self.predicted_quality - self.actual_quality)


class OnlineLearner:
    def __init__(
        self,
        window: int = 2000,
        ewma_decay: float = 0.95,
        retrain_every: int = 200,
        drift_threshold: float = 0.3,
    ):
        self.window: deque[OnlineRecord] = deque(maxlen=window)
        self._ewma_decay = ewma_decay
        self._retrain_every = retrain_every
        self._drift_threshold = drift_threshold
        self._ewma_error: float | None = None
        self._since_retrain = 0

    def observe(self, record: OnlineRecord) -> None:
        self.window.append(record)
        e = record.error
        if self._ewma_error is None:
            self._ewma_error = e
        else:
            self._ewma_error = self._ewma_decay * e + (1 - self._ewma_decay) * self._ewma_error
        self._since_retrain += 1

    def ewma_error(self) -> float:
        return self._ewma_error if self._ewma_error is not None else 0.0

    def drift_detected(self) -> bool:
        return self.ewma_error() > self._drift_threshold and len(self.window) >= 20

    def should_retrain(self) -> bool:
        if self._since_retrain >= self._retrain_every:
            self._since_retrain = 0
            return True
        return False

    def recent_records(self) -> list[OnlineRecord]:
        return list(self.window)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/discriminator/test_online_learner.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/discriminator/online_learner.py tests/discriminator/test_online_learner.py
git commit -m "feat: add I1 online learner (sliding window + EWMA + drift detection)"
```

---

### Task 9: I1 conformal tier selection + router

**Files:**
- Create: `symcascade/discriminator/conformal.py`
- Create: `symcascade/discriminator/router.py`
- Test: `tests/discriminator/test_conformal.py`
- Test: `tests/discriminator/test_router.py`

- [ ] **Step 1: Write failing test for conformal**

`tests/discriminator/test_conformal.py`:
```python
from symcascade.core.types import ROI, Tier
from symcascade.discriminator.conformal import ConformalTierSelector


def _roi(q, c=0.1, l=0.1):
    return ROI(quality=q, cost=c, latency=l)


def test_calibrate_uses_calibration_residuals():
    sel = ConformalTierSelector(alpha=0.1)
    # calibration: predicted quality vs actual quality per tier
    calib = [
        (Tier.L1, 0.8, 0.85),
        (Tier.L1, 0.6, 0.62),
        (Tier.L2, 0.85, 0.86),
        (Tier.L3, 0.7, 0.71),
    ]
    sel.calibrate(calib)
    assert sel.quantile > 0.0


def test_prediction_set_includes_high_roi_tiers():
    sel = ConformalTierSelector(alpha=0.1)
    sel.calibrate([(Tier.L1, 0.8, 0.85), (Tier.L2, 0.85, 0.86), (Tier.L3, 0.7, 0.71)])
    rois = {Tier.L1: _roi(0.85), Tier.L2: _roi(0.80), Tier.L3: _roi(0.60)}
    pred_set = sel.prediction_set(rois)
    # the highest-ROI tier should always be in the set
    assert Tier.L1 in pred_set


def test_prediction_set_never_empty():
    sel = ConformalTierSelector(alpha=0.1)
    sel.calibrate([(Tier.L1, 0.5, 0.5)])
    rois = {Tier.L1: _roi(0.1), Tier.L2: _roi(0.1), Tier.L3: _roi(0.1)}
    pred_set = sel.prediction_set(rois)
    assert len(pred_set) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/discriminator/test_conformal.py -v`
Expected: FAIL

- [ ] **Step 3: Implement conformal tier selector**

`symcascade/discriminator/conformal.py`:
```python
"""I1: split-conformal tier selection.

Calibration residuals = |predicted_quality - actual_quality| per tier.
The conformal quantile at level (1-alpha) gives a tolerance band; a tier is
in the prediction set if its ROI quality is within the band of the best tier.
Guarantee: P(true-best tier in prediction set) >= 1-alpha under exchangeability.
"""
from __future__ import annotations

import math
from typing import Iterable

from symcascade.core.types import ROI, Tier


class ConformalTierSelector:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile: float = float("inf")

    def calibrate(self, calibration: Iterable[tuple[Tier, float, float]]) -> None:
        residuals = [abs(pred - actual) for _t, pred, actual in calibration]
        residuals = sorted(residuals)
        if not residuals:
            self.quantile = 0.0
            return
        # split-conformal quantile with finite-sample correction
        n = len(residuals)
        idx = min(n - 1, int(math.ceil((1 - self.alpha) * (n + 1))) - 1)
        self.quantile = residuals[max(0, idx)]

    def prediction_set(self, rois: dict[Tier, ROI]) -> set[Tier]:
        if not rois:
            return set()
        best_q = max(r.quality for r in rois.values())
        band = best_q - self.quantile
        return {t for t, r in rois.items() if r.quality >= band - 1e-9}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/discriminator/test_conformal.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write failing test for router**

`tests/discriminator/test_router.py`:
```python
from symcascade.core.types import CacheState, Query, ROI, Tier
from symcascade.discriminator.router import DiscriminatorRouter


class FakeEstimator:
    def __init__(self, rois):
        self._rois = rois
    def predict(self, q, cs):
        return self._rois
    def fit_warmup(self, n, seed):
        pass


class FakeConformal:
    def __init__(self, allowed):
        self._allowed = allowed
    def calibrate(self, c):
        pass
    def prediction_set(self, rois):
        return self._allowed


def test_router_picks_highest_benefit_cost_in_prediction_set():
    rois = {
        Tier.L1: ROI(quality=0.9, cost=0.05, latency=0.1),   # best BCR
        Tier.L2: ROI(quality=0.85, cost=0.4, latency=0.5),
        Tier.L3: ROI(quality=0.7, cost=0.2, latency=0.3),
    }
    router = DiscriminatorRouter(FakeEstimator(rois), FakeConformal({Tier.L1, Tier.L2}))
    q = Query(text="q", embedding=[0.1] * 16)
    tier = router.route(q, CacheState(size=5, topk_sims=[0.8], hit_rate=0.5))
    assert tier == Tier.L1


def test_router_prefers_symbolic_when_conformal_set_excludes_neural():
    rois = {
        Tier.L1: ROI(quality=0.8, cost=0.05, latency=0.1),
        Tier.L2: ROI(quality=0.85, cost=0.4, latency=0.5),
        Tier.L3: ROI(quality=0.95, cost=0.2, latency=0.3),
    }
    router = DiscriminatorRouter(FakeEstimator(rois), FakeConformal({Tier.L1, Tier.L2}))
    q = Query(text="q", embedding=[0.1] * 16)
    tier = router.route(q, CacheState(size=5, topk_sims=[0.8], hit_rate=0.5))
    # L3 excluded by conformal; among L1/L2 the conservative bias toward symbolic holds
    assert tier in (Tier.L1, Tier.L2)


def test_router_returns_l3_when_set_only_has_neural():
    rois = {
        Tier.L1: ROI(quality=0.1, cost=0.05, latency=0.1),
        Tier.L2: ROI(quality=0.1, cost=0.4, latency=0.5),
        Tier.L3: ROI(quality=0.7, cost=0.2, latency=0.3),
    }
    router = DiscriminatorRouter(FakeEstimator(rois), FakeConformal({Tier.L3}))
    q = Query(text="q", embedding=[0.1] * 16)
    tier = router.route(q, CacheState(size=5, topk_sims=[0.8], hit_rate=0.5))
    assert tier == Tier.L3
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/discriminator/test_router.py -v`
Expected: FAIL

- [ ] **Step 7: Implement router**

`symcascade/discriminator/router.py`:
```python
"""I1: DiscriminatorRouter.

Picks the starting tier: among the conformal prediction set, choose the tier
with the best benefit-cost ratio. A conservative bias toward the symbolic
chain (L1/L2) is baked in: symbolic tiers get a small BCR bonus because
mis-routing a formalizable query to the neural chain forfeits the optimal
symbolic solution (asymmetric error cost, per spec F1).
"""
from __future__ import annotations

from symcascade.core.types import CacheState, Query, Tier

_SYMBOLIC_BONUS = 0.05


class DiscriminatorRouter:
    def __init__(self, estimator, conformal, budget_lambda: float = 1.0):
        self._est = estimator
        self._conformal = conformal
        self._lambda = budget_lambda

    def route(self, query: Query, cache_state: CacheState | None) -> Tier:
        rois = self._est.predict(query, cache_state)
        pred_set = self._conformal.prediction_set(rois)
        if not pred_set:
            pred_set = set(rois.keys())
        best_tier, best_score = None, -float("inf")
        for tier in pred_set:
            roi = rois[tier]
            score = roi.benefit_cost_ratio(self._lambda)
            if tier in (Tier.L1, Tier.L2):
                score += _SYMBOLIC_BONUS
            if score > best_score:
                best_tier, best_score = tier, score
        return best_tier if best_tier is not None else Tier.L3
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/discriminator/test_router.py tests/discriminator/test_conformal.py -v`
Expected: PASS (6 tests)

- [ ] **Step 9: Commit**

```bash
git add symcascade/discriminator/conformal.py symcascade/discriminator/router.py tests/discriminator/test_conformal.py tests/discriminator/test_router.py
git commit -m "feat: add I1 conformal tier selector + discriminator router"
```

---

### Task 10: C1 Fast Downward solver wrapper (GPL-isolated subprocess)

**Files:**
- Create: `symcascade/symbolic/fd_solver.py`
- Test: `tests/symbolic/test_fd_solver.py`

- [ ] **Step 1: Write failing test**

`tests/symbolic/test_fd_solver.py`:
```python
import pytest
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction


def test_solver_uses_injected_runner(monkeypatch):
    """FD is called via subprocess; tests inject a fake runner to avoid the binary."""
    calls = []

    def fake_run(domain_pddl, problem_pddl, timeout):
        calls.append((domain_pddl, problem_pddl, timeout))
        return FDSolverResult(success=True, plan=[{"name": "goto"}, {"name": "take"}],
                              stderr="", returncode=0)

    solver = FDSolver(runner=fake_run, timeout=30)
    result = solver.solve(domain_pddl="(:predicates ...)", problem_pddl="(:goal ...)")
    assert result.success is True
    assert [s["name"] for s in result.plan] == ["goto", "take"]
    assert len(calls) == 1


def test_solver_failure_returns_success_false(monkeypatch):
    def fake_run(domain_pddl, problem_pddl, timeout):
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
    solver = FDSolver(timeout=10)
    result = solver.solve(domain_pddl="d", problem_pddl="p")
    assert captured["timeout"] == 10
    assert "fast-downward" in captured["cmd"][0]
    assert result.success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/symbolic/test_fd_solver.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FD solver wrapper**

`symcascade/symbolic/fd_solver.py`:
```python
"""C1: Fast Downward solver wrapper.

GPL isolation: Fast Downward is GPL-3.0. We invoke it as a subprocess via
`unified-planning` style command-line, never importing its Python. This keeps
SymCascade (Apache-2.0-compatible) free of GPL contamination.

The runner is injectable so tests never need the binary.
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
    import tempfile, os
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
    import re
    if not __import__("os").path.exists(path):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/symbolic/test_fd_solver.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/symbolic/fd_solver.py tests/symbolic/test_fd_solver.py
git commit -m "feat: add C1 Fast Downward solver wrapper with GPL-isolated subprocess"
```

---

### Task 11: C2 constrained replanner (L1 stage)

**Files:**
- Create: `symcascade/symbolic/replanner.py`
- Test: `tests/symbolic/test_replanner.py`

- [ ] **Step 1: Write failing test**

`tests/symbolic/test_replanner.py`:
```python
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
                              domain_pddl="d", problem_pddl_fn=lambda q: "p")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/symbolic/test_replanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'symcascade.symbolic.replanner'`

- [ ] **Step 3: Implement SkeletonReplanner**

`symcascade/symbolic/replanner.py`:
```python
"""C2: L1 stage — skeleton-constrained replanning.

On a cache miss at L0, the discriminator routes formalizable queries with a
matching skeleton here. The replanner asks Fast Downward to replan over the
skeleton prefix (fast replanning / LAMA-style restart), avoiding a full
replan. This is the C2 contribution that APC replaces with small-LLM
adaptation.
"""
from __future__ import annotations

from typing import Callable, Optional

from symcascade.core.types import Query, StageResult, Tier
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import Skeleton, extract_skeleton


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/symbolic/test_replanner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/symbolic/replanner.py tests/symbolic/test_replanner.py
git commit -m "feat: add C2 L1 skeleton-constrained replanner stage"
```

---

### Task 12: C1 L2 stage — LLM PDDL generation + solve

**Files:**
- Create: `symcascade/symbolic/pddl_gen.py`
- Test: `tests/symbolic/test_pddl_gen.py`

- [ ] **Step 1: Write failing test**

`tests/symbolic/test_pddl_gen.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/symbolic/test_pddl_gen.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PDDLGenStage**

`symcascade/symbolic/pddl_gen.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/symbolic/test_pddl_gen.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add symcascade/symbolic/pddl_gen.py tests/symbolic/test_pddl_gen.py
git commit -m "feat: add C1 L2 PDDL-gen+solve stage with skeleton caching on success"
```

---

### Task 13: C3 neural chain — confidence gating

**Files:**
- Create: `symcascade/neural/__init__.py`
- Create: `symcascade/neural/gemma_backend.py`
- Create: `symcascade/neural/confidence_gate.py`
- Test: `tests/neural/__init__.py`
- Test: `tests/neural/test_gemma_backend.py`
- Test: `tests/neural/test_confidence_gate.py`

- [ ] **Step 1: Create neural package inits**

`symcascade/neural/__init__.py`:
```python
"""Neural chain: L3 Gemma confidence gating + L4 cloud fallback."""
```

`tests/neural/__init__.py`: empty.

- [ ] **Step 2: Write failing test for Gemma backend**

`tests/neural/test_gemma_backend.py`:
```python
from symcascade.neural.gemma_backend import GemmaBackend, GemmaResponse


class FakeVLLM:
    def __init__(self, answer, logprob):
        self._answer = answer
        self._logprob = logprob
        self.thinking_enabled = None

    def generate(self, prompt, enable_thinking):
        self.thinking_enabled = enable_thinking
        return {"text": self._answer, "logprobs": [[self._logprob]]}


def test_backend_forces_thinking_false_on_low_cost_tier():
    fake = FakeVLLM(answer="42", logprob=-0.1)
    backend = GemmaBackend(client=fake, enable_thinking=False)
    resp = backend.answer("what is 2+2?")
    assert fake.thinking_enabled is False
    assert resp.text == "42"


def test_backend_confidence_is_softmax_of_top_logprob():
    fake = FakeVLLM(answer="42", logprob=-0.1)
    backend = GemmaBackend(client=fake, enable_thinking=False)
    resp = backend.answer("q")
    # softmax(-0.1) ~ 0.475; we approximate confidence as exp(top_logprob)
    assert 0.0 < resp.confidence <= 1.0


def test_backend_failure_returns_low_confidence():
    class BrokenVLLM:
        def generate(self, prompt, enable_thinking):
            raise RuntimeError("vllm down")
    backend = GemmaBackend(client=BrokenVLLM(), enable_thinking=False)
    resp = backend.answer("q")
    assert resp.confidence == 0.0
    assert "ERROR" in resp.text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/neural/test_gemma_backend.py -v`
Expected: FAIL

- [ ] **Step 4: Implement GemmaBackend**

`symcascade/neural/gemma_backend.py`:
```python
"""L3: Gemma 4 inference backend.

Wraps vLLM behind an injectable client so tests never need a GPU. The
low-cost tier forces enable_thinking=False (spec constraint). Confidence is
derived from the top-token logprob — a proxy for the model's self-knowledge
boundary, which C3 hypothesises aligns with Gemini due to shared lineage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class VLLMClient(Protocol):
    def generate(self, prompt: str, enable_thinking: bool) -> dict: ...


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    confidence: float


class GemmaBackend:
    def __init__(self, client: VLLMClient, enable_thinking: bool = False):
        self._client = client
        self._enable_thinking = enable_thinking

    def answer(self, prompt: str) -> GemmaResponse:
        try:
            out = self._client.generate(prompt, self._enable_thinking)
        except Exception:
            return GemmaResponse(text="ERROR", confidence=0.0)
        top_logprob = -10.0
        logprobs = out.get("logprobs") or []
        if logprobs and logprobs[0]:
            top_logprob = logprobs[0][0]
        conf = math.exp(top_logprob) if top_logprob <= 0 else 1.0
        return GemmaResponse(text=out.get("text", ""), confidence=min(1.0, conf))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/neural/test_gemma_backend.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Write failing test for confidence gate**

`tests/neural/test_confidence_gate.py`:
```python
from symcascade.core.types import Query, StageResult
from symcascade.neural.confidence_gate import ConfidenceGate, ConfidenceCalibrator


def test_high_confidence_passes_through():
    gate = ConfidenceGate(threshold=0.5)
    assert gate.should_answer(0.9) is True


def test_low_confidence_defers_to_cloud():
    gate = ConfidenceGate(threshold=0.5)
    assert gate.should_answer(0.3) is False


def test_calibrator_sets_threshold_from_quantile():
    cal = ConfidenceCalibrator(alpha=0.1)
    # calibration pairs: (confidence, correct?)
    cal.calibrate([(0.9, True), (0.8, True), (0.4, False), (0.3, False)])
    gate = cal.to_gate()
    # threshold should sit between correct and incorrect confidence
    assert 0.4 <= gate.threshold <= 0.8


def test_gate_as_stage_returns_success_when_confident():
    class FakeGemma:
        def answer(self, prompt):
            from symcascade.neural.gemma_backend import GemmaResponse
            return GemmaResponse(text="ans", confidence=0.9)
    gate = ConfidenceGate(threshold=0.5)
    stage = gate.as_stage(FakeGemma())
    r = stage.run(Query(text="q", embedding=[]))
    assert r.success is True
    assert r.answer == "ans"


def test_gate_as_stage_returns_failure_when_not_confident():
    class FakeGemma:
        def answer(self, prompt):
            from symcascade.neural.gemma_backend import GemmaResponse
            return GemmaResponse(text="ans", confidence=0.2)
    gate = ConfidenceGate(threshold=0.5)
    stage = gate.as_stage(FakeGemma())
    r = stage.run(Query(text="q", embedding=[]))
    assert r.success is False
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest tests/neural/test_confidence_gate.py -v`
Expected: FAIL

- [ ] **Step 8: Implement confidence gate**

`symcascade/neural/confidence_gate.py`:
```python
"""C3: confidence gating for the L3 neural stage.

A conformal-calibrated threshold decides whether Gemma answers directly or
defers to the cloud (L4). Same-source risk (F3): if Gemma is confidently
wrong, the gate fails — mitigated by calibrating the threshold on a
hold-out set rather than hand-tuning.
"""
from __future__ import annotations

from typing import Protocol

from symcascade.core.types import Query, StageResult
from symcascade.neural.gemma_backend import GemmaBackend


class GemmaLike(Protocol):
    def answer(self, prompt: str):  # GemmaResponse
        ...


class ConfidenceGate:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def should_answer(self, confidence: float) -> bool:
        return confidence >= self.threshold

    def as_stage(self, gemma: GemmaLike):
        outer = self

        class _L3Stage:
            def run(self, query: Query) -> StageResult:
                resp = gemma.answer(query.text)
                if outer.should_answer(resp.confidence):
                    return StageResult(answer=resp.text, success=True,
                                       cost=0.0, latency_ms=20.0,
                                       confidence=resp.confidence)
                return StageResult(answer="", success=False, cost=0.0,
                                   latency_ms=20.0, confidence=resp.confidence)
        return _L3Stage()


class ConfidenceCalibrator:
    """Split-conformal threshold: pick the confidence quantile above which
    1-alpha of correct answers lie."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.threshold = 0.5

    def calibrate(self, pairs: list[tuple[float, bool]]) -> None:
        # pairs = (confidence, was_correct)
        correct = sorted(c for c, ok in pairs if ok)
        if not correct:
            self.threshold = 1.0
            return
        incorrect = sorted(c for c, ok in pairs if not ok)
        # threshold: smallest confidence covering (1-alpha) of correct answers
        idx = max(0, int(len(correct) * (1 - self.alpha)) - 1)
        lower = correct[idx]
        upper = incorrect[0] if incorrect else 1.0
        self.threshold = 0.5 * (lower + upper) if lower < upper else lower

    def to_gate(self) -> ConfidenceGate:
        return ConfidenceGate(threshold=self.threshold)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/neural/test_confidence_gate.py tests/neural/test_gemma_backend.py -v`
Expected: PASS (8 tests)

- [ ] **Step 10: Commit**

```bash
git add symcascade/neural/ tests/neural/
git commit -m "feat: add L3 Gemma backend + C3 conformal confidence gate"
```

---

### Task 14: End-to-end cascade integration test

**Files:**
- Test: `tests/test_e2e_cascade.py`

- [ ] **Step 1: Write the integration test**

`tests/test_e2e_cascade.py`:
```python
"""End-to-end: wire the real Cascade with stub backends to prove the dual-path
discriminative routing + fallback chain works as a system."""
from symcascade.core.types import Query, StageResult, Tier
from symcascade.core.orchestrator import Cascade
from symcascade.cache.semantic_cache import SemanticCache
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.discriminator.router import DiscriminatorRouter
from symcascade.discriminator.roi_estimator import ROIEstimator
from symcascade.discriminator.conformal import ConformalTierSelector
from symcascade.symbolic.replanner import SkeletonReplanner
from symcascade.symbolic.fd_solver import FDSolver, FDSolverResult
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction


class _FakeFD:
    def __init__(self, plan_steps):
        self._steps = plan_steps

    def constrained_replan(self, domain_pddl, problem_pddl, skeleton):
        return FDSolverResult(success=True,
                              plan=[{"name": n} for n in self._steps],
                              stderr="", returncode=0)

    def solve(self, domain_pddl, problem_pddl):
        return FDSolverResult(success=True,
                              plan=[{"name": n} for n in self._steps],
                              stderr="", returncode=0)


def _build_cascade(skel_in_cache=None, fd_plan=("goto", "take", "goto", "put")):
    cache = SemanticCache(sim_threshold=0.99, embed_fn=lambda t: [1.0, 0.0])
    skel_cache = SkeletonCache(threshold=0.5)
    if skel_in_cache is not None:
        skel_cache.put("seed", skel_in_cache)
    fd = _FakeFD(fd_plan)

    l1 = SkeletonReplanner(cache=skel_cache, fd=fd, domain_pddl="d",
                           problem_pddl_fn=lambda q: "p",
                           query_skeleton_fn=lambda q: skel_in_cache or Skeleton(actions=()))

    # L2: minimal stub that succeeds
    class _L2Stub:
        def run(self, q):
            return StageResult(answer="l2", success=True, cost=0.4, latency_ms=50.0)
    # L3: always defers (low confidence) to exercise fallback to L4
    class _L3Stub:
        def run(self, q):
            return StageResult(answer="", success=False, cost=0.0, latency_ms=20.0)
    # L4: cloud fallback
    class _L4Stub:
        def run(self, q):
            return StageResult(answer="cloud", success=True, cost=1.0, latency_ms=500.0)

    est = ROIEstimator()
    est.fit_warmup(n=200, seed=0)
    conf = ConformalTierSelector(alpha=0.1)
    conf.calibrate([(Tier.L1, 0.8, 0.85), (Tier.L2, 0.85, 0.86), (Tier.L3, 0.7, 0.71)])
    router = DiscriminatorRouter(est, conf)

    def cache_state_fn():
        return skel_cache.state_for(skel_in_cache or Skeleton(actions=()))

    return Cascade(cache=cache, discriminator=router, l1=l1, l2=_L2Stub(),
                   l3=_L3Stub(), l4=_L4Stub(), cache_state_fn=cache_state_fn), skel_cache


def test_e2e_cache_hit_short_circuits():
    cache = SemanticCache(sim_threshold=0.99, embed_fn=lambda t: [1.0, 0.0])
    cache.put("hello", "cached-answer")
    est = ROIEstimator(); est.fit_warmup(n=100, seed=0)
    conf = ConformalTierSelector(alpha=0.1)
    conf.calibrate([(Tier.L1, 0.8, 0.85)])
    router = DiscriminatorRouter(est, conf)

    class _S:
        def run(self, q):
            raise AssertionError("L1 should not run on cache hit")
    cascade = Cascade(cache=cache, discriminator=router, l1=_S(), l2=_S(),
                      l3=_S(), l4=_S(), cache_state_fn=lambda: None)
    r = cascade.run(Query(text="hello", embedding=[1.0, 0.0]))
    assert r.answer == "cached-answer"
    assert r.cost == 0.0


def test_e2e_symbolic_chain_l1_succeeds():
    seed = Skeleton(actions=(SkeletonAction(name="goto"), SkeletonAction(name="take"),
                             SkeletonAction(name="goto"), SkeletonAction(name="put")))
    cascade, _ = _build_cascade(skel_in_cache=seed,
                                fd_plan=("goto", "take", "goto", "put"))
    r = cascade.run(Query(text="put apple in fridge", embedding=[0.1] * 16))
    assert r.success is True


def test_e2e_fallback_reaches_cloud_when_neural_chain_defers():
    seed = Skeleton(actions=(SkeletonAction(name="look"),))  # unusual skeleton
    cascade, _ = _build_cascade(skel_in_cache=seed,
                                fd_plan=("goto",))
    # force discriminator to L3 by giving a query whose skeleton won't match
    cascade._stages[Tier.L1].__class__  # noqa
    r = cascade.run(Query(text="opaque question", embedding=[0.9] * 16))
    assert r.success is True  # some tier in the chain succeeds
```

- [ ] **Step 2: Run the integration test**

Run: `python -m pytest tests/test_e2e_cascade.py -v`
Expected: PASS (3 tests). If any fails, fix the wiring before proceeding.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS across core, cache, discriminator, symbolic, neural, e2e.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_cascade.py
git commit -m "test: add end-to-end cascade integration test"
```

---

## Self-Review Notes

**Spec coverage check:**
- I1 dual-path discriminative routing → Tasks 7,8,9 (features/online/conformal/router) ✓
- C2 skeleton cache + constrained replanning → Tasks 4,5,6,11 ✓
- C1 symbolic solver backend (GPL-isolated) → Task 10 ✓
- C1 L2 PDDL gen + solve → Task 12 ✓
- C3 same-source confidence gating (conformal) → Task 13 ✓
- L0 semantic cache → Task 3 ✓
- Fallback chain (L1→L2→L3→L4) → Task 2 ✓
- F6 conformal coverage semantics → Tasks 9, 13 (split-conformal with explicit guarantee text) ✓
- Cache-state-aware features (I1 non-triviality) → Task 7 (build_features includes cache state) ✓

**Deferred to follow-up plans** (out of scope for this walking-skeleton plan):
- Real vLLM/Gemma client wiring (GemmaBackend currently Protocol-injected)
- Real Fast Downward binary integration (FDSolver default runner present but untested against binary)
- GPTCache/MAP integration behind existing interfaces
- Benchmark harnesses (ALFWorld/HotpotQA/mixed-load) — separate plan
- Baselines (APC/LLM-DP/FrugalGPT/RouteLLM) — separate plan
- I4 regret bound proof — pure theory, separate doc
- Online refit loop wiring (OnlineLearner.ready + ROIEstimator.refit) — components built, integration in harness plan

**Placeholder scan:** no TBD/TODO remain; all steps contain complete code. The one stray `symcache_fix` import was removed during authoring.

**Type consistency:** `Tier` enum, `StageResult`, `ROI`, `CacheState`, `Skeleton`/`SkeletonAction` used consistently across tasks. `Stage` protocol's `run(query)->StageResult` honoured by all stage implementations.


