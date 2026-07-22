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

OUTPUT_COLUMNS = ["geometry", "confidence", "area_sqm"]


def empty_result() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in OUTPUT_COLUMNS}, geometry="geometry", crs=4326)


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

    # Same engine as geoai.regularize (buildingregulariser), same defaults as
    # the geoai building-footprint example. Preserves attribute columns.
    gdf = regularize_geodataframe(
        gdf,
        parallel_threshold=1.0,
        simplify=True,
        simplify_tolerance=0.5,
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

    gdf = gdf.to_crs(4326)
    # Keep whole footprints that touch the AOI rather than slicing them at the
    # edge — the raster was already clipped, so nothing extends far beyond it.
    gdf = gdf[gdf.geometry.intersects(box(*bbox))]
    if gdf.empty:
        return empty_result()

    return gdf[OUTPUT_COLUMNS].reset_index(drop=True)
