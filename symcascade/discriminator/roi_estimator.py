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
