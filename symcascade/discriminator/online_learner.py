"""I1: online learner for the discriminator.

Sliding window of (tier, predicted_quality, actual_quality) observations.
EWMA-smoothed prediction error feeds drift detection; when error exceeds the
threshold the orchestrator can fall back to a static policy. should_retrain()
paces discriminator refits to avoid thrashing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from symcascade.core.types import Tier


@dataclass(frozen=True)
class OnlineRecord:
    tier: Tier
    predicted_quality: float
    actual_quality: float

    @property
    def error(self) -> float:
        return abs(self.predicted_quality - self.actual_quality)


class OnlineLearner:
    def __init__(
        self,
        window: int = 2000,
        ewma_decay: float = 0.95,
        retrain_every: int = 200,
        drift_threshold: float = 0.3,
    ):
        self.window: deque[OnlineRecord] = deque(maxlen=window)
        self._ewma_decay = ewma_decay
        self._retrain_every = retrain_every
        self._drift_threshold = drift_threshold
        self._ewma_error: float | None = None
        self._since_retrain = 0

    def observe(self, record: OnlineRecord) -> None:
        self.window.append(record)
        e = record.error
        if self._ewma_error is None:
            self._ewma_error = e
        else:
            self._ewma_error = self._ewma_decay * e + (1 - self._ewma_decay) * self._ewma_error
        self._since_retrain += 1

    def ewma_error(self) -> float:
        return self._ewma_error if self._ewma_error is not None else 0.0

    def drift_detected(self) -> bool:
        return self.ewma_error() > self._drift_threshold and len(self.window) >= 20

    def should_retrain(self) -> bool:
        if self._since_retrain >= self._retrain_every:
            self._since_retrain = 0
            return True
        return False

    def recent_records(self) -> list[OnlineRecord]:
        return list(self.window)
