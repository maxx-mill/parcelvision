"""Stage 3: raw detections -> clean, regularized, georeferenced footprints.

Runs entirely on geometry (no ML deps) so it is unit-testable in the slim
image. CRS discipline: input arrives in the raster's projected CRS (or any
CRS with `crs` set); areas are computed in projected meters; output is always
EPSG:4326.
"""

import geopandas as gpd
from buildingregulariser import regularize_geodataframe
from shapely.geometry import box

# Smallest footprint worth keeping, in m². NAIP at 0.6 m can't resolve sheds
# much smaller than this, and sub-20 m² blobs are almost always noise.
MIN_AREA_SQM = 20.0

# Duplicate-footprint thresholds. Backends run their own tile-overlap NMS, but a
# final universal pass catches cross-tile/cross-raster leftovers AND the case
# NMS-by-IoU misses: a small blob nested inside a real building (low IoU, high
# containment). Adjacent row-houses share an edge but barely overlap, so neither
# rule merges them.
DEDUP_IOU = 0.6
DEDUP_CONTAINMENT = 0.8

OUTPUT_COLUMNS = ["geometry", "confidence", "area_sqm"]


def empty_result() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in OUTPUT_COLUMNS}, geometry="geometry", crs=4326)


def dedup_overlaps(
    gdf: gpd.GeoDataFrame, iou_thr: float = DEDUP_IOU, contain_thr: float = DEDUP_CONTAINMENT
) -> gpd.GeoDataFrame:
    """Drop near-duplicate footprints. Processing largest-first, a polygon is a
    duplicate if it overlaps an already-kept one by IoU > iou_thr (same building
    twice) or if that much of *its own* area sits inside a kept one (a nested
    blob). Keeps the larger footprint. Pure geometry — slim-testable."""
    if len(gdf) < 2:
        return gdf
    order = gdf.geometry.area.sort_values(ascending=False).index
    gdf = gdf.loc[order].reset_index(drop=True)
    sindex = gdf.sindex
    kept_pos: list[int] = []
    is_dup = [False] * len(gdf)
    for i, g in enumerate(gdf.geometry):
        if g.is_empty or g.area <= 0:
            is_dup[i] = True
            continue
        for j in sindex.query(g, predicate="intersects"):
            if j >= i or is_dup[j]:  # only compare against already-kept, larger polys
                continue
            kg = gdf.geometry.iloc[j]
            inter = g.intersection(kg).area
            if inter <= 0:
                continue
            iou = inter / (g.area + kg.area - inter)
            if iou > iou_thr or inter / g.area > contain_thr:
                is_dup[i] = True
                break
        if not is_dup[i]:
            kept_pos.append(i)
    return gdf.iloc[kept_pos].reset_index(drop=True)


def postprocess(gdf: gpd.GeoDataFrame | None, bbox: list[float]) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return empty_result()
    if gdf.crs is None:
        raise ValueError("detections have no CRS — refusing to guess (see CRS discipline)")

    # Regularization needs metric coordinates; reproject geographic input to UTM.
    if not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    if "confidence" not in gdf.columns:
        gdf["confidence"] = None

    # Regularize raw masks into clean building outlines (buildingregulariser,
    # the engine geoai.regularize wraps). Params tuned on leaf-off residential
    # detections from the default RF-DETR backend (its masks are rounder than
    # Mask R-CNN's — median ~45 raw vertices): these cut that to ~6 per polygon
    # with no area loss. simplify_tolerance strips stair-stepping before
    # squaring; parallel_threshold merges near-parallel segments; 45° stays on
    # so L-shaped homes fit without the orthogonal-only over-shrink (10-14%
    # area loss in testing).
    gdf = regularize_geodataframe(
        gdf,
        parallel_threshold=3.0,
        simplify=True,
        simplify_tolerance=1.5,
        allow_45_degree=True,
        allow_circles=False,  # buildings, not storage tanks
    )

    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        return empty_result()

    gdf["area_sqm"] = gdf.geometry.area.round(1)  # projected CRS -> true m²
    gdf = gdf[gdf["area_sqm"] >= MIN_AREA_SQM]
    if gdf.empty:
        return empty_result()

    # Universal duplicate removal (backend-agnostic; catches nested/cross-raster
    # duplicates the per-tile IoU NMS leaves behind).
    gdf = dedup_overlaps(gdf)

    gdf = gdf.to_crs(4326)
    # Keep whole footprints that touch the AOI rather than slicing them at the
    # edge — the raster was already clipped, so nothing extends far beyond it.
    gdf = gdf[gdf.geometry.intersects(box(*bbox))]
    if gdf.empty:
        return empty_result()

    return gdf[OUTPUT_COLUMNS].reset_index(drop=True)
