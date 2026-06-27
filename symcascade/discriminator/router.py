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
