import pytest
from app.schemas import JobCreate, bbox_area_km2
from pydantic import ValidationError

STL_BBOX = [-90.31, 38.61, -90.30, 38.62]  # ~1 km² near Clayton, MO


def test_valid_bbox_accepted():
    assert JobCreate(bbox=STL_BBOX).bbox == STL_BBOX


@pytest.mark.parametrize(
    "bbox",
    [
        [-90.30, 38.61, -90.31, 38.62],  # min_lon > max_lon
        [-90.31, 38.62, -90.30, 38.61],  # min_lat > max_lat
        [-190.0, 38.61, -90.30, 38.62],  # lon out of range
        [-90.31, 38.61, -90.30, 95.0],  # lat out of range
        [-90.31, 38.61, -90.30],  # wrong length
    ],
)
def test_invalid_bbox_rejected(bbox):
    with pytest.raises(ValidationError):
        JobCreate(bbox=bbox)


def test_bbox_area_equator_degree_square():
    # 1°×1° at the equator ≈ 110.57 × 111.32 km
    area = bbox_area_km2([0, -0.5, 1, 0.5])
    assert area == pytest.approx(110.574 * 111.320 * 0.99996, rel=0.01)


def test_bbox_area_stl_sample():
    # ~0.87 km × 1.11 km at 38.6°N
    assert bbox_area_km2(STL_BBOX) == pytest.approx(0.97, abs=0.05)
