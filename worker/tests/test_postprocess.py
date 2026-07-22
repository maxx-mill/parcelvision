import geopandas as gpd
import pytest
from shapely.geometry import Point, box
from worker.pipeline.postprocess import MIN_AREA_SQM, postprocess

# UTM 15N (St. Louis). A bbox in 4326 roughly matching the UTM geometries below.
UTM = "EPSG:26915"
BBOX = [-90.32, 38.64, -90.30, 38.66]


def utm_box(cx: float, cy: float, w: float, h: float):
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def make_gdf(geoms, crs=UTM, confidence=0.9):
    return gpd.GeoDataFrame({"confidence": [confidence] * len(geoms)}, geometry=geoms, crs=crs)


def center_utm():
    # Center of BBOX projected to UTM 15N.
    pt = gpd.GeoSeries([Point(-90.31, 38.65)], crs=4326).to_crs(UTM).iloc[0]
    return pt.x, pt.y


def test_output_is_4326_with_true_metric_areas():
    x, y = center_utm()
    gdf = postprocess(make_gdf([utm_box(x, y, 20, 10)]), BBOX)
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326
    # 20 m × 10 m building: area survives regularization within tolerance
    assert gdf.iloc[0]["area_sqm"] == pytest.approx(200, rel=0.1)
    assert gdf.iloc[0]["confidence"] == 0.9


def test_slivers_below_min_area_dropped():
    x, y = center_utm()
    gdf = postprocess(make_gdf([utm_box(x, y, 4, 4), utm_box(x + 100, y, 20, 10)]), BBOX)
    assert len(gdf) == 1
    assert all(gdf["area_sqm"] >= MIN_AREA_SQM)


def test_detections_outside_bbox_dropped():
    x, y = center_utm()
    far = utm_box(x + 50_000, y, 20, 10)  # ~50 km east, well outside bbox
    gdf = postprocess(make_gdf([utm_box(x, y, 20, 10), far]), BBOX)
    assert len(gdf) == 1


def test_empty_and_none_input():
    for raw in (None, gpd.GeoDataFrame(geometry=[], crs=UTM)):
        out = postprocess(raw, BBOX)
        assert out.empty and out.crs.to_epsg() == 4326


def test_missing_crs_rejected():
    x, y = center_utm()
    with pytest.raises(ValueError, match="CRS"):
        postprocess(gpd.GeoDataFrame(geometry=[utm_box(x, y, 20, 10)]), BBOX)


def test_geographic_input_reprojected_not_trusted_for_area():
    # Same building handed over in 4326 — area must still come out in m².
    x, y = center_utm()
    gdf_utm = make_gdf([utm_box(x, y, 20, 10)])
    gdf_4326 = gdf_utm.to_crs(4326)
    out = postprocess(gdf_4326, BBOX)
    assert out.iloc[0]["area_sqm"] == pytest.approx(200, rel=0.1)
