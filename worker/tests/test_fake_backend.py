from shapely.geometry import box
from worker.backends import get_backend

BBOX = [-90.32, 38.64, -90.31, 38.65]


def test_fake_backend_deterministic_and_inside_bbox():
    be = get_backend("fake")
    assert be.needs_imagery is False
    a = be.extract([], BBOX)
    b = be.extract([], BBOX)
    assert len(a) == len(b) > 10
    assert a.geometry.geom_equals(b.geometry).all()
    aoi = box(*BBOX)
    assert a.to_crs(4326).geometry.within(aoi.buffer(1e-6)).all()


def test_unknown_backend_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        get_backend("quantum")
