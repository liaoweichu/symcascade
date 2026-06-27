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
