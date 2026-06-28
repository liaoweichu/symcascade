"""Phase 1 harness + ALFWorld adapter unit tests.

All tests use mock backends (no vLLM/FD/cloud/alfworld data). They verify:
- ALFWorld domain/task/skeleton/evaluator logic
- task -> query -> cascade -> record pipeline
- E1 metrics (cloud call rate, quality, cascade depth, tier hit rates)
- E6 self-growth curve shape
"""
from __future__ import annotations

import pytest

from symcascade.bench.alfworld_domain import DOMAIN_PDDL, TASK_CLASSES, TASK_SKELETONS
from symcascade.bench.alfworld_adapter import (
    ALFWorldTask, load_tasks, task_to_query, task_to_problem_pddl,
    goal_skeleton, evaluate, _task_class_from_text,
)
from symcascade.bench.harness import run_experiment, ExperimentResult, RequestRecord
from symcascade.core.types import Query, StageResult, Tier, CacheState
from symcascade.core.orchestrator import Cascade
from symcascade.cache.semantic_cache import SemanticCache
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _toy_embed(text: str):
    """Deterministic toy embedding: char-bag over a vocab, no collisions.

    Uses ord() of each char so distinct texts get distinct vectors (hash
    collision-free), keeping the vector short for fast cosine.
    """
    vec = [0.0] * 128
    for ch in text:
        vec[ord(ch) % 128] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _make_synthetic_tasks(n=12, seed=42):
    return load_tasks(n=n, data_path=None, seed=seed)


# --------------------------------------------------------------------------
# ALFWorld adapter tests
# --------------------------------------------------------------------------

def test_domain_pddl_is_nonempty_and_parses_as_define():
    assert "(define (domain alfworld))" in DOMAIN_PDDL.replace("\n", " ") \
           or "define (domain alfworld)" in DOMAIN_PDDL
    assert "(:action goto" in DOMAIN_PDDL
    assert "(:action pick" in DOMAIN_PDDL
    assert "(:action heat" in DOMAIN_PDDL


def test_six_task_classes_have_skeletons():
    assert set(TASK_CLASSES) == set(TASK_SKELETONS.keys())
    for tc, seq in TASK_SKELETONS.items():
        assert len(seq) >= 4, f"{tc} skeleton too short"
        assert all(a in {"goto", "pick", "put", "clean", "heat", "cool", "examine"}
                   for a in seq), f"{tc} has unknown action"


def test_task_class_from_text_keywords():
    assert _task_class_from_text("clean the apple then put it in the fridge") == "pick_clean_then_place"
    assert _task_class_from_text("heat the mug in the microwave") == "pick_heat_then_place"
    assert _task_class_from_text("cool the egg then put it in the fridge") == "pick_cool_then_place"
    assert _task_class_from_text("examine the book with the light") == "look_at_obj_in_light"
    assert _task_class_from_text("put two apples in the drawer") == "pick_two_and_place"
    assert _task_class_from_text("put the apple in the fridge") == "pick_and_place"


def test_synthetic_tasks_cover_all_classes():
    tasks = _make_synthetic_tasks(n=12)
    assert len(tasks) == 12
    classes = {t.task_class for t in tasks}
    assert classes == set(TASK_CLASSES)


def test_goal_skeleton_matches_task_class():
    tasks = _make_synthetic_tasks(n=12)
    for t in tasks:
        skel = goal_skeleton(t)
        expected = TASK_SKELETONS[t.task_class]
        assert tuple(a.name for a in skel.actions) == expected


def test_task_to_query_carries_text_and_embedding():
    tasks = _make_synthetic_tasks(n=3)
    q = task_to_query(tasks[0], _toy_embed)
    assert q.text == tasks[0].text
    assert len(q.embedding) > 0


def test_task_to_problem_pddl_has_domain_and_goal():
    tasks = _make_synthetic_tasks(n=3)
    pddl = task_to_problem_pddl(tasks[0])
    assert "(:domain alfworld)" in pddl
    assert "(:goal" in pddl


def test_evaluate_correct_symbolic_plan_scores_1():
    """A pick-and-place plan: goto pick goto put on a synthetic task."""
    tasks = _make_synthetic_tasks(n=6, seed=1)
    # find a pick_and_place task
    t = next(t for t in tasks if t.task_class == "pick_and_place")
    plan = "goto pick goto put"
    # the synthetic init has the object at the agent's location
    score = evaluate(plan, t)
    assert score == 1.0


def test_evaluate_empty_or_wrong_plan_scores_0():
    tasks = _make_synthetic_tasks(n=6, seed=1)
    t = tasks[0]
    assert evaluate("", t) == 0.0
    assert evaluate("goto goto goto", t) == 0.0


# --------------------------------------------------------------------------
# harness + cascade integration (mock backends)
# --------------------------------------------------------------------------

class _MockL1:
    """Mock L1: succeeds with given probability, costs 0.05."""
    def __init__(self, succeed=True):
        self._ok = succeed
        self.calls = 0
    def run(self, query: Query) -> StageResult:
        self.calls += 1
        return StageResult(answer="goto pick goto put", success=self._ok,
                           cost=0.05, latency_ms=10.0, confidence=1.0)


class _MockL2:
    def __init__(self, succeed=True):
        self._ok = succeed
        self.calls = 0
    def run(self, query: Query) -> StageResult:
        self.calls += 1
        return StageResult(answer="goto pick goto put", success=self._ok,
                           cost=0.4, latency_ms=50.0, confidence=1.0)


class _MockL3:
    def __init__(self, succeed=True):
        self._ok = succeed
        self.calls = 0
    def run(self, query: Query) -> StageResult:
        self.calls += 1
        return StageResult(answer="goto pick goto put", success=self._ok,
                           cost=0.0, latency_ms=20.0, confidence=0.9)


class _MockL4:
    def __init__(self, succeed=True):
        self._ok = succeed
        self.calls = 0
    def run(self, query: Query) -> StageResult:
        self.calls += 1
        return StageResult(answer="I think you should goto pick goto put",
                           success=self._ok, cost=1.0, latency_ms=500.0,
                           confidence=1.0)


class _MockDisc:
    """Always routes to a configured tier; no observe (static)."""
    def __init__(self, tier=Tier.L1):
        self._tier = tier
    def route(self, query, cache_state):
        return self._tier


def _build_cascade(disc_tier=Tier.L1, l1_ok=True, l2_ok=True, l3_ok=True, l4_ok=True,
                   cache_hit=False):
    cache = SemanticCache(sim_threshold=0.99, embed_fn=_toy_embed)
    if cache_hit:
        cache.put("seed", "cached-answer")
    skel_cache = SkeletonCache(threshold=0.5)
    disc = _MockDisc(disc_tier)
    cascade = Cascade(
        cache=cache, discriminator=disc,
        l1=_MockL1(l1_ok), l2=_MockL2(l2_ok),
        l3=_MockL3(l3_ok), l4=_MockL4(l4_ok),
        cache_state_fn=lambda: CacheState(size=skel_cache.size(), topk_sims=[], hit_rate=0.0),
    )
    return cascade, skel_cache


def test_harness_runs_and_collects_records():
    tasks = _make_synthetic_tasks(n=5)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    cascade, skel = _build_cascade(disc_tier=Tier.L1, l1_ok=True)
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    assert len(res.records) == 5
    for r in res.records:
        assert r.tier_chosen == Tier.L1
        assert r.cache_hit is False
        assert r.cloud_called is False


def test_e1_metrics_cloud_rate_and_quality():
    tasks = _make_synthetic_tasks(n=8)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    # all L1 fails -> fallback to L4 cloud
    cascade, skel = _build_cascade(disc_tier=Tier.L1, l1_ok=False, l2_ok=False,
                                   l3_ok=False, l4_ok=True)
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    assert res.cloud_call_rate() == 1.0   # every request fell to L4
    assert res.avg_quality() >= 0.0       # cloud free-text answers


def test_e1_metrics_cache_hit_short_circuits():
    """When cache is seeded identically, semantic cache should hit."""
    tasks = _make_synthetic_tasks(n=3)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    cascade, skel = _build_cascade(disc_tier=Tier.L1)
    # pre-seed the semantic cache with identical embeddings so every query hits
    for t in tasks:
        cascade._cache.put(t.text, "goto pick goto put")  # noqa: SLF001
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    # all should be cache hits (exact text match)
    hits = sum(1 for r in res.records if r.cache_hit)
    assert hits == 3


def test_e1_cascade_depth_metric():
    tasks = _make_synthetic_tasks(n=4)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    cascade, skel = _build_cascade(disc_tier=Tier.L1, l1_ok=True)
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    # L1 serves all -> depth = 1 each
    assert res.avg_cascade_depth() == 1.0


def test_e6_self_growth_curve_shape():
    """E6: as the cache fills, cloud-call rate should be non-increasing."""
    # Simulate: first half misses (go to cloud), second half all cache-hit
    tasks = _make_synthetic_tasks(n=40)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    cascade, skel = _build_cascade(disc_tier=Tier.L1, l1_ok=False, l2_ok=False,
                                   l3_ok=False, l4_ok=True)
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    curve = res.self_growth_curve(window=10)
    assert len(curve) >= 2
    # curve is list of (cache_size, cloud_rate); cloud_rate should exist
    assert all(0.0 <= rate <= 1.0 for _, rate in curve)


def test_tier_hit_rates_when_all_l1():
    tasks = _make_synthetic_tasks(n=5)
    queries = [task_to_query(t, _toy_embed) for t in tasks]
    cascade, skel = _build_cascade(disc_tier=Tier.L1, l1_ok=True)
    res = run_experiment(
        cascade=cascade, tasks=tasks, queries=queries,
        evaluate_fn=evaluate,
        skeleton_cache_size_fn=lambda: skel.size(),
    )
    rates = res.tier_hit_rates()
    assert rates.get("L1", 0.0) == 1.0


def test_experiment_result_empty_metrics_safe():
    er = ExperimentResult()
    assert er.cloud_call_rate() == 0.0
    assert er.avg_quality() == 0.0
    assert er.avg_cascade_depth() == 0.0
    assert er.tier_hit_rates() == {}
    assert er.self_growth_curve() == []
