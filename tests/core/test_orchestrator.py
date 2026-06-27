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
