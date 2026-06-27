from symcascade.core.types import ROI, Tier
from symcascade.discriminator.conformal import ConformalTierSelector


def _roi(q, c=0.1, l=0.1):
    return ROI(quality=q, cost=c, latency=l)


def test_calibrate_uses_calibration_residuals():
    sel = ConformalTierSelector(alpha=0.1)
    # calibration: predicted quality vs actual quality per tier
    calib = [
        (Tier.L1, 0.8, 0.85),
        (Tier.L1, 0.6, 0.62),
        (Tier.L2, 0.85, 0.86),
        (Tier.L3, 0.7, 0.71),
    ]
    sel.calibrate(calib)
    assert sel.quantile > 0.0


def test_prediction_set_includes_high_roi_tiers():
    sel = ConformalTierSelector(alpha=0.1)
    sel.calibrate([(Tier.L1, 0.8, 0.85), (Tier.L2, 0.85, 0.86), (Tier.L3, 0.7, 0.71)])
    rois = {Tier.L1: _roi(0.85), Tier.L2: _roi(0.80), Tier.L3: _roi(0.60)}
    pred_set = sel.prediction_set(rois)
    # the highest-ROI tier should always be in the set
    assert Tier.L1 in pred_set


def test_prediction_set_never_empty():
    sel = ConformalTierSelector(alpha=0.1)
    sel.calibrate([(Tier.L1, 0.5, 0.5)])
    rois = {Tier.L1: _roi(0.1), Tier.L2: _roi(0.1), Tier.L3: _roi(0.1)}
    pred_set = sel.prediction_set(rois)
    assert len(pred_set) >= 1
