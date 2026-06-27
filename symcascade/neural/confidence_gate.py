"""C3: confidence gating for the L3 neural stage.

A conformal-calibrated threshold decides whether Gemma answers directly or
defers to the cloud (L4). Same-source risk (F3): if Gemma is confidently
wrong, the gate fails — mitigated by calibrating the threshold on a
hold-out set rather than hand-tuning.
"""
from __future__ import annotations

from typing import Protocol

from symcascade.core.types import Query, StageResult


class GemmaLike(Protocol):
    def answer(self, prompt: str):  # GemmaResponse
        ...


class ConfidenceGate:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def should_answer(self, confidence: float) -> bool:
        return confidence >= self.threshold

    def as_stage(self, gemma: GemmaLike):
        outer = self

        class _L3Stage:
            def run(self, query: Query) -> StageResult:
                resp = gemma.answer(query.text)
                if outer.should_answer(resp.confidence):
                    return StageResult(answer=resp.text, success=True,
                                       cost=0.0, latency_ms=20.0,
                                       confidence=resp.confidence)
                return StageResult(answer="", success=False, cost=0.0,
                                   latency_ms=20.0, confidence=resp.confidence)
        return _L3Stage()


class ConfidenceCalibrator:
    """Split-conformal threshold: pick the confidence quantile above which
    1-alpha of correct answers lie."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.threshold = 0.5

    def calibrate(self, pairs: list[tuple[float, bool]]) -> None:
        # pairs = (confidence, was_correct)
        correct = sorted(c for c, ok in pairs if ok)
        if not correct:
            self.threshold = 1.0
            return
        incorrect = sorted(c for c, ok in pairs if not ok)
        # threshold: smallest confidence covering (1-alpha) of correct answers
        idx = max(0, int(len(correct) * (1 - self.alpha)) - 1)
        lower = correct[idx]
        upper = incorrect[0] if incorrect else 1.0
        self.threshold = 0.5 * (lower + upper) if lower < upper else lower

    def to_gate(self) -> ConfidenceGate:
        return ConfidenceGate(threshold=self.threshold)
