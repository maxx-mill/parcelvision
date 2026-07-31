"""Unit tests for the property-report summary rollup (Chapter 7)."""

from app.routers.parcels import summarize_structures


def _s(condition, area=100.0):
    return {"condition": condition, "area_sqm": area, "crosses_boundary": False}


def test_empty_is_ok():
    s = summarize_structures([])
    assert s["structure_count"] == 0
    assert s["worst_condition"] == "ok"
    assert s["total_building_area_sqm"] == 0


def test_worst_condition_includes_damaged():
    # regression: the v5 model emits "damaged"; a collapsed roof must not report "ok".
    s = summarize_structures([_s("ok"), _s("review"), _s("damaged")])
    assert s["worst_condition"] == "damaged"
    assert s["condition_counts"]["damaged"] == 1


def test_severity_order_tarp_beats_damaged():
    s = summarize_structures([_s("damaged"), _s("tarp"), _s("review")])
    assert s["worst_condition"] == "tarp"


def test_damaged_beats_review():
    s = summarize_structures([_s("review"), _s("damaged")])
    assert s["worst_condition"] == "damaged"


def test_area_sums_and_rounds_none_safe():
    s = summarize_structures([_s("ok", 10.04), _s("ok", None), _s("ok", 5.0)])
    assert s["total_building_area_sqm"] == 15.0
    assert s["structure_count"] == 3
