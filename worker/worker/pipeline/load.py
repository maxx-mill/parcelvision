"""Stage 4: write footprints to PostGIS.

Table contract mirrors api/app/models.py (Building); the services share the
schema, not code — see that module's docstring.
"""

import uuid

import geopandas as gpd
from sqlalchemy import Engine, text


def load_buildings(engine: Engine, job_id: str, gdf: gpd.GeoDataFrame) -> int:
    # Idempotent under RQ retries: a rerun replaces the job's rows, never appends twice.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM buildings WHERE job_id = :jid"), {"jid": job_id})
    if gdf.empty:
        return 0
    out = gdf.rename_geometry("geom").copy()
    out["job_id"] = [uuid.UUID(job_id)] * len(out)  # UUID objects so psycopg types the column
    out[["geom", "job_id", "confidence", "area_sqm"]].to_postgis(
        "buildings", engine, if_exists="append", index=False
    )
    return len(out)
