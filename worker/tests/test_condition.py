import numpy as np
from worker.pipeline.condition import flag, tarp_fraction


def _fill(color, n=400):
    return np.tile(np.array(color, dtype="uint8"), (n, 1))


def test_blue_tarp_detected():
    assert tarp_fraction(_fill([40, 60, 200])) > 0.9


def test_blue_grey_winter_roof_not_tarp():
    # winter blue-grey cast: blue only marginally above r/g -> must NOT flag
    assert tarp_fraction(_fill([130, 135, 150])) == 0.0


def test_brown_shingle_not_tarp():
    assert tarp_fraction(_fill([110, 80, 60])) == 0.0


def test_flag_tarp_wins():
    # a tarp fraction over threshold flags 'tarp' regardless of score
    assert flag(0.9, 0.2) == "tarp"


def test_flag_damaged_review_ok_by_score():
    # thresholds: DAMAGED >= 0.60, REVIEW >= 0.50 (calibrated to v5 scores)
    assert flag(0.8, 0.0) == "damaged"
    assert flag(0.55, 0.0) == "review"
    assert flag(0.45, 0.0) == "ok"


def test_flag_no_score_is_ok():
    assert flag(None, 0.0) == "ok"


def test_partial_tarp_above_threshold():
    roof = np.vstack([_fill([40, 60, 200], 60), _fill([120, 110, 100], 240)])  # 20% tarp
    assert tarp_fraction(roof) >= 0.10
    assert flag(0.0, tarp_fraction(roof)) == "tarp"
