"""Chapter 3: load authoritative county parcels for an AOI and validate
detected footprints against them.

Parcels come from St. Louis County's open ArcGIS REST service (owner-neutral
fields only). We fetch the parcels intersecting a job's bbox, upsert them by
locator, then run PostGIS spatial joins to answer the questions the whole
project is really about:

  * how many detected buildings sit on each parcel,
  * which footprints cross a parcel boundary (detection or parcel-line error),
  * which parcels have no detected structure.
"""

import logging
import os

import geopandas as gpd
import requests
from shapely.geometry import box
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

# Configurable so the loader isn't hard-wired to one county (see .env.example).
PARCEL_SERVICE_URL = os.environ.get(
    "PARCEL_SERVICE_URL",
    "https://maps.stlouisco.com/hosting/rest/services/Maps/AGS_Parcels/MapServer/0",
)
PARCEL_LOCATOR_FIELD = os.environ.get("PARCEL_LOCATOR_FIELD", "LOCATOR")
PARCEL_ADDRESS_FIELD = os.environ.get("PARCEL_ADDRESS_FIELD", "PROP_ADD")
PAGE_SIZE = 2000
REQUEST_TIMEOUT = 60


def fetch_parcels(bbox: list[float]) -> gpd.GeoDataFrame:
    """Page through the ArcGIS REST layer for parcels intersecting bbox."""
    params = {
        "where": "1=1",
        "geometry": ",".join(str(c) for c in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": f"{PARCEL_LOCATOR_FIELD},{PARCEL_ADDRESS_FIELD}",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": PAGE_SIZE,
    }
    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    while True:
        resp = requests.get(
            f"{PARCEL_SERVICE_URL}/query",
            params={**params, "resultOffset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        fc = resp.json()
        feats = fc.get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs=4326))
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not frames:
        return gpd.GeoDataFrame(
            {"locator": [], "address": []}, geometry=[], crs=4326
        )
    import pandas as pd

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    gdf = gdf.rename(
        columns={PARCEL_LOCATOR_FIELD: "locator", PARCEL_ADDRESS_FIELD: "address"}
    )
    gdf = gdf[gdf.geometry.notna() & gdf["locator"].notna()]
    gdf = gdf[gdf.geometry.intersects(box(*bbox))]
    # Layer can return the same parcel twice across pages at page boundaries.
    gdf = gdf.drop_duplicates(subset="locator")
    return gdf[["locator", "address", "geometry"]]


def upsert_parcels(engine: Engine, gdf: gpd.GeoDataFrame) -> int:
    """Insert parcels new to us; existing locators keep their row. Uses a temp
    table + ON CONFLICT so re-validating overlapping AOIs never duplicates."""
    if gdf.empty:
        return 0
    out = gdf.rename_geometry("geom").copy()
    out.to_postgis("parcels_stage", engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO parcels (locator, address, geom) "
                "SELECT locator, address, ST_SetSRID(geom, 4326) FROM parcels_stage "
                "ON CONFLICT (locator) DO NOTHING"
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS parcels_stage"))
    return len(out)


def validate_job(engine: Engine, job_id: str, bbox: list[float]) -> dict:
    """Spatial-join the job's buildings against parcels in the AOI. Returns a
    summary; per-parcel detail is served live by the API from the same joins."""
    envelope = "ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)"
    params = {
        "jid": job_id,
        "minx": bbox[0],
        "miny": bbox[1],
        "maxx": bbox[2],
        "maxy": bbox[3],
    }
    with engine.begin() as conn:
        row = conn.execute(
            text(
                f"""
                WITH aoi_parcels AS (
                    SELECT id, geom FROM parcels
                    WHERE ST_Intersects(geom, {envelope})
                ),
                job_buildings AS (
                    SELECT id, geom, ST_PointOnSurface(geom) AS pt
                    FROM buildings WHERE job_id = :jid
                ),
                -- a building belongs to the parcel containing its interior point
                assigned AS (
                    SELECT b.id AS bid, p.id AS pid
                    FROM job_buildings b
                    LEFT JOIN aoi_parcels p ON ST_Contains(p.geom, b.pt)
                )
                SELECT
                    (SELECT count(*) FROM aoi_parcels) AS parcels_total,
                    (SELECT count(DISTINCT pid) FROM assigned WHERE pid IS NOT NULL)
                        AS parcels_with_buildings,
                    (SELECT count(*) FROM job_buildings) AS buildings_total,
                    (SELECT count(*) FROM assigned WHERE pid IS NULL)
                        AS buildings_off_parcel,
                    (SELECT count(*) FROM job_buildings b
                        WHERE (SELECT count(*) FROM aoi_parcels p
                               WHERE ST_Overlaps(p.geom, b.geom)) > 0)
                        AS buildings_crossing
                """
            ),
            params,
        ).one()
    summary = {
        "parcels_total": row.parcels_total,
        "parcels_with_buildings": row.parcels_with_buildings,
        "parcels_empty": row.parcels_total - row.parcels_with_buildings,
        "buildings_total": row.buildings_total,
        "buildings_off_parcel": row.buildings_off_parcel,
        "buildings_crossing": row.buildings_crossing,
    }
    logger.info("job %s validation: %s", job_id, summary)
    return summary
