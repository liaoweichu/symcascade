"""Tests for the online discriminator router: route + observe + refit + drift."""
from symcascade.core.types import CacheState, Query, ROI, StageResult, Tier
from symcascade.discriminator.online_router import OnlineDiscriminatorRouter
from symcascade.discriminator.online_learner import OnlineLearner


class FakeEstimator:
    """Records refit calls; returns configurable ROIs."""
    def __init__(self, rois):
        self._rois = rois
        self.refit_calls = 0
        self.last_refit_data = None

    def predict(self, q, cs):
        return self._rois

    def fit_warmup(self, n, seed):
        pass

    def refit(self, records):
        self.refit_calls += 1
        self.last_refit_data = list(records)


class FakeConformal:
    def __init__(self, allowed):
        self._allowed = allowed
    def calibrate(self, c):
        pass
    def prediction_set(self, rois):
        return self._allowed


def _q():
    return Query(text="q", embedding=[0.1] * 16)


def _cs():
    return CacheState(size=5, topk_sims=[0.8], hit_rate=0.5)


def _rois():
    return {
        Tier.L1: ROI(quality=0.9, cost=0.05, latency=0.1),
        Tier.L2: ROI(quality=0.85, cost=0.4, latency=0.5),
        Tier.L3: ROI(quality=0.7, cost=0.2, latency=0.3),
    }


def test_route_returns_tier_and_stashes_predicted_quality():
    est = FakeEstimator(_rois())
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1, Tier.L2}),
                                       online=OnlineLearner(retrain_every=1000))
    tier = router.route(_q(), _cs())
    assert tier == Tier.L1
    assert router.last_predicted_quality() == 0.9  # L1's predicted quality


def test_observe_feeds_online_learner_without_refit_below_period():
    est = FakeEstimator(_rois())
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1}),
                                       online=OnlineLearner(retrain_every=5))
    router.route(_q(), _cs())
    for _ in range(4):
        router.observe(0.9)
        assert est.refit_calls == 0
    router.observe(0.9)
    assert est.refit_calls == 1
    assert est.last_refit_data is not None


def test_observe_triggers_refit_with_recent_records():
    est = FakeEstimator(_rois())
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1}),
                                       online=OnlineLearner(retrain_every=3, window=100))
    # produce a few route/observe pairs
    for actual in [0.85, 0.90, 0.80]:
        router.route(_q(), _cs())
        router.observe(actual)
    # retrain_every=3 -> one refit after the 3rd observe
    assert est.refit_calls == 1
    records = est.last_refit_data
    assert len(records) == 3
    # each record carries tier + predicted + actual
    assert all(r.tier == Tier.L1 for r in records)


def test_drift_triggers_static_fallback():
    est = FakeEstimator(_rois())
    online = OnlineLearner(retrain_every=1000, ewma_decay=0.9, drift_threshold=0.3)
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1}),
                                       online=online, fallback_tier=Tier.L3)
    # baseline: predictions match actuals (no drift)
    for _ in range(20):
        router.route(_q(), _cs())   # predicts L1 quality 0.9
        router.observe(0.9)
    assert not router.in_drift()
    assert router.route(_q(), _cs()) == Tier.L1
    # distribution shift: actual quality collapses while prediction stays 0.9
    for _ in range(20):
        router.route(_q(), _cs())
        router.observe(0.1)
    assert router.in_drift()
    # in drift, router returns the fallback tier instead of the ROI-chosen one
    assert router.route(_q(), _cs()) == Tier.L3


def test_drift_recovers_when_error_falls_back_below_threshold():
    est = FakeEstimator(_rois())
    online = OnlineLearner(retrain_every=1000, ewma_decay=0.9, drift_threshold=0.3)
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1}),
                                       online=online, fallback_tier=Tier.L3)
    # trigger drift
    for _ in range(25):
        router.route(_q(), _cs())
        router.observe(0.1)
    assert router.in_drift()
    # recover: predictions now match actuals again
    for _ in range(30):
        router.route(_q(), _cs())
        router.observe(0.9)
    assert not router.in_drift()
    assert router.route(_q(), _cs()) == Tier.L1


def test_observe_without_route_is_safe():
    est = FakeEstimator(_rois())
    router = OnlineDiscriminatorRouter(estimator=est, conformal=FakeConformal({Tier.L1}),
                                       online=OnlineLearner())
    # observe before any route -> no-op, no crash
    router.observe(0.5)
    assert est.refit_calls == 0
