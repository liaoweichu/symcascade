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


class _RouterOverride:
    """Wraps a router but forces a chosen tier, for deterministic e2e paths."""
    def __init__(self, tier):
        self._tier = tier
    def route(self, query, cache_state):
        return self._tier


def _build_cascade(skel_in_cache=None, fd_plan=("goto", "take", "goto", "put"),
                   forced_tier=None):
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

    if forced_tier is not None:
        router = _RouterOverride(forced_tier)
    else:
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
                                fd_plan=("goto", "take", "goto", "put"),
                                forced_tier=Tier.L1)
    r = cascade.run(Query(text="put apple in fridge", embedding=[0.1] * 16))
    assert r.success is True
    assert r.answer == "goto take goto put"


def test_e2e_fallback_reaches_cloud_when_neural_chain_defers():
    seed = Skeleton(actions=(SkeletonAction(name="look"),))  # unusual skeleton
    cascade, _ = _build_cascade(skel_in_cache=seed,
                                fd_plan=("goto",),
                                forced_tier=Tier.L3)
    # L1/L2 skipped (forced L3), L3 defers -> falls back to L4 cloud
    r = cascade.run(Query(text="opaque question", embedding=[0.9] * 16))
    assert r.success is True
    assert r.answer == "cloud"


def test_e2e_l1_miss_l2_caches_skeleton_self_growth():
    """Forced L1 with no matching skeleton -> falls back to L2, which succeeds.
    Demonstrates the fallback chain L1->L2."""
    cascade, skel_cache = _build_cascade(skel_in_cache=None,
                                         forced_tier=Tier.L1)
    r = cascade.run(Query(text="new task", embedding=[0.1] * 16))
    assert r.success is True
    assert r.answer == "l2"  # L2 stub answered after L1 miss


# ---------------------------------------------------------------------------
# Self-growth loop: OnlineDiscriminatorRouter + Cascade observer hook.
# ---------------------------------------------------------------------------

class _FakeEstimator:
    """Returns a fixed ROI set; records refit calls."""
    def __init__(self, rois):
        self._rois = rois
        self.refit_calls = 0
    def predict(self, q, cs):
        return self._rois
    def fit_warmup(self, n, seed):
        pass
    def refit(self, records):
        self.refit_calls += 1


class _FakeConformal:
    def __init__(self, allowed):
        self._allowed = allowed
    def calibrate(self, c):
        pass
    def prediction_set(self, rois):
        return self._allowed


class _QualityStage:
    """Stage whose output quality (confidence) can be mutated mid-test."""
    def __init__(self, quality=0.9, success=True):
        self.quality = quality
        self.success = success
        self.calls = 0
    def run(self, q):
        self.calls += 1
        return StageResult(answer="ok", success=self.success,
                           cost=0.0, latency_ms=1.0, confidence=self.quality)


class _RecordingCache:
    """No-op cache (always miss) so every run reaches the discriminator."""
    def get(self, query):
        return None
    def put(self, query, answer):
        pass


def _self_growth_rois():
    from symcascade.core.types import ROI
    return {
        Tier.L1: ROI(quality=0.9, cost=0.05, latency=0.1),
        Tier.L2: ROI(quality=0.85, cost=0.4, latency=0.5),
        Tier.L3: ROI(quality=0.7, cost=0.2, latency=0.3),
        Tier.L4: ROI(quality=0.95, cost=1.0, latency=2.0),
    }


def _build_self_growth_cascade(l1_quality=0.9, retrain_every=5,
                               drift_threshold=0.3):
    from symcascade.discriminator.online_router import OnlineDiscriminatorRouter
    from symcascade.discriminator.online_learner import OnlineLearner
    est = _FakeEstimator(_self_growth_rois())
    online = OnlineLearner(retrain_every=retrain_every,
                           ewma_decay=0.9,
                           drift_threshold=drift_threshold)
    router = OnlineDiscriminatorRouter(
        estimator=est,
        conformal=_FakeConformal({Tier.L1}),
        online=online,
        fallback_tier=Tier.L3,
    )
    l1 = _QualityStage(quality=l1_quality)
    l3 = _QualityStage(quality=0.7)
    cascade = Cascade(
        cache=_RecordingCache(),
        discriminator=router,
        l1=l1,
        l2=_QualityStage(quality=0.85),
        l3=l3,
        l4=_QualityStage(quality=0.95),
        cache_state_fn=lambda: None,
        quality_fn=lambda r: r.confidence,
    )
    return cascade, router, est, l1, l3


def test_e2e_self_growth_refit_triggers_on_observations():
    """Observer hook feeds observe() -> OnlineLearner -> refit() after
    retrain_every observations. Proves the loop is closed end-to-end."""
    cascade, router, est, l1, l3 = _build_self_growth_cascade(retrain_every=5)
    # 6 distinct queries (cache always misses) at quality 0.9
    for i in range(6):
        cascade.run(Query(text=f"q{i}", embedding=[float(i + 1)] * 16))
    assert est.refit_calls >= 1
    assert not router.in_drift()
    assert l1.calls == 6
    assert l3.calls == 0  # never fell back


def test_e2e_self_growth_drift_switches_to_fallback_tier():
    """When quality collapses globally, the online learner detects drift and
    the router serves L3 (fallback) instead of L1. Proves drift fallback works
    end-to-end through the Cascade.

    Note: the shift must be global (all tiers degrade) so that drift persists
    after the fallback kicks in — if only L1 degraded, L3's stable quality
    would match its prediction, the EWMA would drop, and drift would clear
    immediately. A global shift models a real distribution drift.
    """
    cascade, router, est, l1, l3 = _build_self_growth_cascade(retrain_every=1000)
    # Warm up: 20 successful runs at quality 0.9 (no drift)
    for i in range(20):
        cascade.run(Query(text=f"warm{i}", embedding=[float(i + 1)] * 16))
    assert not router.in_drift()
    # Distribution shift: ALL tiers' quality collapses to 0.1
    l1.quality = 0.1
    l3.quality = 0.1
    for i in range(20, 50):
        cascade.run(Query(text=f"shift{i}", embedding=[float(i + 1)] * 16))
    assert router.in_drift()
    # In drift, route() returns L3 -> L1 should not be called for new queries
    l1_calls_at_drift = l1.calls
    cascade.run(Query(text="post-drift", embedding=[0.5] * 16))
    assert l1.calls == l1_calls_at_drift  # L1 not called while in drift
    assert l3.calls >= 1  # L3 (fallback) served the post-drift query


def test_e2e_self_growth_drift_recovers_when_quality_restores():
    """After drift, if quality restores to match predictions, the EWMA drops
    back below threshold and drift clears — proving self-healing.

    While in drift, route() serves L3 and pairs L3's prediction (0.7) with
    L3's actual. When L3's quality restores to 0.7, error drops to 0, the
    EWMA falls below threshold, and drift clears.
    """
    cascade, router, est, l1, l3 = _build_self_growth_cascade(retrain_every=1000)
    # Trigger drift via global quality collapse
    l1.quality = 0.1
    l3.quality = 0.1
    for i in range(30):
        cascade.run(Query(text=f"bad{i}", embedding=[float(i + 1)] * 16))
    assert router.in_drift()
    # Restore: L3 quality back to 0.7 (matches L3's prediction).
    # L1 quality also restored for post-recovery routing.
    l1.quality = 0.9
    l3.quality = 0.7
    for i in range(30):
        cascade.run(Query(text=f"recover{i}", embedding=[float(i + 1)] * 16))
    assert not router.in_drift()
    # Post-recovery: route() returns the estimator's choice (L1) again
    l1_calls_before = l1.calls
    cascade.run(Query(text="post-recover", embedding=[0.5] * 16))
    assert l1.calls == l1_calls_before + 1  # L1 served again
