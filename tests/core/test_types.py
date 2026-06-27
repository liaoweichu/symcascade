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
