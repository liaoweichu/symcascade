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
