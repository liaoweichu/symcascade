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
