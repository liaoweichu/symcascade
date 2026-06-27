from symcascade.core.types import Query, StageResult
from symcascade.neural.confidence_gate import ConfidenceGate, ConfidenceCalibrator


def test_high_confidence_passes_through():
    gate = ConfidenceGate(threshold=0.5)
    assert gate.should_answer(0.9) is True


def test_low_confidence_defers_to_cloud():
    gate = ConfidenceGate(threshold=0.5)
    assert gate.should_answer(0.3) is False


def test_calibrator_sets_threshold_from_quantile():
    cal = ConfidenceCalibrator(alpha=0.1)
    # calibration pairs: (confidence, correct?)
    cal.calibrate([(0.9, True), (0.8, True), (0.4, False), (0.3, False)])
    gate = cal.to_gate()
    # threshold should sit between correct and incorrect confidence
    assert 0.4 <= gate.threshold <= 0.8


def test_gate_as_stage_returns_success_when_confident():
    class FakeGemma:
        def answer(self, prompt):
            from symcascade.neural.gemma_backend import GemmaResponse
            return GemmaResponse(text="ans", confidence=0.9)
    gate = ConfidenceGate(threshold=0.5)
    stage = gate.as_stage(FakeGemma())
    r = stage.run(Query(text="q", embedding=[]))
    assert r.success is True
    assert r.answer == "ans"


def test_gate_as_stage_returns_failure_when_not_confident():
    class FakeGemma:
        def answer(self, prompt):
            from symcascade.neural.gemma_backend import GemmaResponse
            return GemmaResponse(text="ans", confidence=0.2)
    gate = ConfidenceGate(threshold=0.5)
    stage = gate.as_stage(FakeGemma())
    r = stage.run(Query(text="q", embedding=[]))
    assert r.success is False
