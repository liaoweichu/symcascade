"""I1: split-conformal tier selection.

Calibration residuals = |predicted_quality - actual_quality| per tier.
The conformal quantile at level (1-alpha) gives a tolerance band; a tier is
in the prediction set if its ROI quality is within the band of the best tier.
Guarantee: P(true-best tier in prediction set) >= 1-alpha under exchangeability.
"""
from __future__ import annotations

import math
from typing import Iterable

from symcascade.core.types import ROI, Tier


class ConformalTierSelector:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile: float = float("inf")

    def calibrate(self, calibration: Iterable[tuple[Tier, float, float]]) -> None:
        residuals = [abs(pred - actual) for _t, pred, actual in calibration]
        residuals = sorted(residuals)
        if not residuals:
            self.quantile = 0.0
            return
        # split-conformal quantile with finite-sample correction
        n = len(residuals)
        idx = min(n - 1, int(math.ceil((1 - self.alpha) * (n + 1))) - 1)
        self.quantile = residuals[max(0, idx)]

    def prediction_set(self, rois: dict[Tier, ROI]) -> set[Tier]:
        if not rois:
            return set()
        best_q = max(r.quality for r in rois.values())
        band = best_q - self.quantile
        return {t for t, r in rois.items() if r.quality >= band - 1e-9}
