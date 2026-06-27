from symcascade.core.types import Tier
from symcascade.discriminator.online_learner import OnlineLearner, OnlineRecord


def _rec(tier=Tier.L1, predicted_q=0.5, actual_q=0.9):
    return OnlineRecord(tier=tier, predicted_quality=predicted_q, actual_quality=actual_q)


def test_records_accumulate_in_window():
    ol = OnlineLearner(window=5)
    for _ in range(3):
        ol.observe(_rec())
    assert len(ol.window) == 3


def test_window_evicts_old_records():
    ol = OnlineLearner(window=2)
    ol.observe(_rec(tier=Tier.L1))
    ol.observe(_rec(tier=Tier.L2))
    ol.observe(_rec(tier=Tier.L3))
    assert len(ol.window) == 2
    assert ol.window[-1].tier == Tier.L3


def test_ewma_error_smooths_predictions():
    ol = OnlineLearner(window=100, ewma_decay=0.5)
    ol.observe(_rec(predicted_q=0.5, actual_q=0.9))  # error 0.4
    assert abs(ol.ewma_error() - 0.4) < 1e-9
    ol.observe(_rec(predicted_q=0.5, actual_q=0.5))  # error 0.0
    # EWMA = 0.5*0.0 + 0.5*0.4 = 0.2
    assert abs(ol.ewma_error() - 0.2) < 1e-9


def test_drift_detected_when_error_jumps():
    ol = OnlineLearner(window=100, ewma_decay=0.9, drift_threshold=0.3)
    # baseline: low error
    for _ in range(20):
        ol.observe(_rec(predicted_q=0.9, actual_q=0.9))
    assert not ol.drift_detected()
    # sudden distribution shift
    for _ in range(20):
        ol.observe(_rec(predicted_q=0.9, actual_q=0.1))
    assert ol.drift_detected()


def test_should_retrain_respects_period():
    ol = OnlineLearner(window=100, retrain_every=5)
    for _ in range(4):
        ol.observe(_rec())
        assert not ol.should_retrain()
    ol.observe(_rec())
    assert ol.should_retrain()
