import numpy as np
from worker.pipeline.condition import assess_pixels, classify, heterogeneity, tarp_fraction


def _fill(color, n=400):
    return np.tile(np.array(color, dtype="uint8"), (n, 1))


def test_blue_tarp_detected():
    tarp = _fill([40, 60, 200])  # strong blue
    assert tarp_fraction(tarp) > 0.9
    assert classify(tarp_fraction(tarp), 0.0) == "tarp"


def test_clean_grey_roof_not_tarp():
    grey = _fill([120, 120, 125])  # blue only marginally above r/g
    assert tarp_fraction(grey) < 0.05
    # uniform -> low heterogeneity -> ok
    assert assess_pixels(grey)["condition"] == "ok"


def test_brown_shingle_not_tarp():
    brown = _fill([110, 80, 60])
    assert tarp_fraction(brown) == 0.0


def test_uniform_roof_low_heterogeneity():
    assert heterogeneity(_fill([100, 100, 100])) == 0.0


def test_mixed_roof_high_heterogeneity_flags_review():
    # half very dark, half very bright -> large luminance spread
    mixed = np.vstack([_fill([15, 15, 15], 200), _fill([240, 240, 240], 200)])
    het = heterogeneity(mixed)
    assert het > 0.50
    assert classify(0.0, het) == "review"


def test_partial_tarp_above_threshold():
    roof = np.vstack([_fill([40, 60, 200], 60), _fill([120, 110, 100], 240)])  # 20% tarp
    assert tarp_fraction(roof) >= 0.10
    assert assess_pixels(roof)["condition"] == "tarp"


def test_blue_grey_winter_roof_not_tarp():
    # winter blue-grey cast: blue only marginally above r/g -> must NOT flag tarp
    assert tarp_fraction(_fill([130, 135, 150])) == 0.0


def test_empty_pixels_safe():
    empty = np.empty((0, 3), dtype="uint8")
    assert assess_pixels(empty) == {"tarp_fraction": 0.0, "heterogeneity": 0.0, "condition": "ok"}
