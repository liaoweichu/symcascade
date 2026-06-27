"""I1: online discriminator router — closes the self-growth loop.

Wraps an ROIEstimator + ConformalTierSelector + OnlineLearner. On each route()
it predicts ROI, picks the tier (reusing DiscriminatorRouter's logic) and
stashes the predicted quality of the chosen tier. observe(actual_quality)
then:

  1. records (tier, predicted, actual) into the OnlineLearner,
  2. when should_retrain() fires, refits the estimator on the recent window,
  3. when drift_detected() fires, switches to a static fallback tier until
     the EWMA error recovers below the drift threshold.

Drift recovery: while in drift we still serve the fallback tier, but we pair
it with the fallback's own predicted quality (from the estimator's ROI set)
so observe() can keep updating the EWMA. When the distribution stabilizes the
error drops, the EWMA falls back below threshold, and drift clears — no probe
mechanism needed.

This realizes the spec's "self-growing scheduler": as the cache fills and
real outcomes arrive, the estimator refits and routing adapts online.
"""
from __future__ import annotations

from typing import Optional

from symcascade.core.types import CacheState, Query, Tier
from symcascade.discriminator.router import DiscriminatorRouter
from symcascade.discriminator.online_learner import OnlineLearner, OnlineRecord


class OnlineDiscriminatorRouter:
    def __init__(
        self,
        estimator,
        conformal,
        online: OnlineLearner,
        fallback_tier: Tier = Tier.L4,
        budget_lambda: float = 1.0,
    ):
        self._est = estimator
        self._conformal = conformal
        self._online = online
        self._fallback_tier = fallback_tier
        self._router = DiscriminatorRouter(estimator, conformal, budget_lambda)
        self._last_predicted: Optional[float] = None
        self._last_tier: Optional[Tier] = None
        self._in_drift: bool = False

    def route(self, query: Query, cache_state: CacheState | None) -> Tier:
        if self._in_drift:
            # Serve the static fallback for safety, but still stash the
            # fallback tier's predicted quality so observe() can update the
            # EWMA and drift can clear when the distribution stabilizes.
            rois = self._est.predict(query, cache_state)
            self._last_tier = self._fallback_tier
            self._last_predicted = (
                rois[self._fallback_tier].quality
                if self._fallback_tier in rois
                else None
            )
            return self._fallback_tier
        return self._estimator_route(query, cache_state)

    def _estimator_route(self, query: Query, cache_state: CacheState | None) -> Tier:
        rois = self._est.predict(query, cache_state)
        tier = self._router.route(query, cache_state)
        self._last_tier = tier
        self._last_predicted = rois[tier].quality if tier in rois else None
        return tier

    def observe(self, actual_quality: float) -> None:
        # No prediction to pair with (e.g. observe before route, or the
        # fallback tier had no ROI entry to pair against).
        if self._last_tier is None or self._last_predicted is None:
            return
        record = OnlineRecord(
            tier=self._last_tier,
            predicted_quality=self._last_predicted,
            actual_quality=actual_quality,
        )
        self._online.observe(record)
        # Refit pacing
        if self._online.should_retrain() and hasattr(self._est, "refit"):
            self._est.refit(self._online.recent_records())
        # Drift hysteresis: enter drift when detected, exit when error recovers
        if self._online.drift_detected():
            if not self._in_drift:
                self._in_drift = True
        elif self._in_drift and self._online.ewma_error() <= self._online._drift_threshold:
            self._in_drift = False

    def in_drift(self) -> bool:
        return self._in_drift

    def last_predicted_quality(self) -> Optional[float]:
        return self._last_predicted
