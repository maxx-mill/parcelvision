"""Chapter 3: parcel validation — the project's differentiator.

Validation is opt-in per completed job: POST enqueues a worker task that loads
county parcels for the AOI and computes spatial joins; GET serves the summary
plus a per-parcel GeoJSON layer (building count + flags) computed live from the
same joins, so the map can shade parcels by how well detections agree with the
cadastre.
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Job
from ..queue import enqueue_validation
from ..schemas import JobOut, ValidationSummary

router = APIRouter(prefix="/jobs", tags=["validation"])


def _get_job_or_404(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/{job_id}/validate", response_model=JobOut, status_code=202)
def start_validation(job_id: uuid.UUID, session: Session = Depends(get_session)) -> Job:
    """Kick off parcel validation for a completed job."""
    job = _get_job_or_404(session, job_id)
    if job.status != "done":
        raise HTTPException(409, detail=f"job is {job.status}; validate after it's done")
    if job.validation_status in ("loading_parcels", "validating"):
        raise HTTPException(409, detail="validation already in progress")
    job.validation_status = "loading_parcels"
    job.validation_error = None
    session.commit()
    enqueue_validation(str(job_id))
    return job


# Per-parcel layer for map shading. A building is assigned to the parcel
# containing its interior point; a footprint "crosses" when it geometrically
# overlaps a parcel boundary.
_PARCELS_SQL = """
WITH aoi AS (
    SELECT ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326) AS env
),
aoi_parcels AS (
    SELECT p.id, p.locator, p.address, p.geom
    FROM parcels p, aoi WHERE ST_Intersects(p.geom, aoi.env)
),
jb AS (
    SELECT id, geom, ST_PointOnSurface(geom) AS pt
    FROM buildings WHERE job_id = :jid
)
SELECT
    ap.locator,
    ap.address,
    ST_AsGeoJSON(ap.geom) AS geometry,
    (SELECT count(*) FROM jb WHERE ST_Contains(ap.geom, jb.pt)) AS building_count,
    (SELECT count(*) FROM jb WHERE ST_Overlaps(ap.geom, jb.geom)) AS crossing_count
FROM aoi_parcels ap
"""

# Building-centric summary — counts each building once, so overlapping-parcel
# slivers can't double-count. Mirrors worker/worker/pipeline/parcels.validate_job.
_SUMMARY_SQL = """
WITH aoi_parcels AS (
    SELECT id, geom FROM parcels
    WHERE ST_Intersects(geom, ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))
),
jb AS (
    SELECT id, geom, ST_PointOnSurface(geom) AS pt
    FROM buildings WHERE job_id = :jid
),
off AS (  -- a building is "off-parcel" when its interior point is in no parcel
    SELECT b.id FROM jb b
    WHERE NOT EXISTS (SELECT 1 FROM aoi_parcels p WHERE ST_Contains(p.geom, b.pt))
)
SELECT
    (SELECT count(*) FROM aoi_parcels) AS parcels_total,
    -- parcel-centric so it matches the map layer's empty/with-buildings shading
    (SELECT count(*) FROM aoi_parcels p
        WHERE EXISTS (SELECT 1 FROM jb b WHERE ST_Contains(p.geom, b.pt)))
        AS parcels_with_buildings,
    (SELECT count(*) FROM jb) AS buildings_total,
    (SELECT count(*) FROM off) AS buildings_off_parcel,
    (SELECT count(*) FROM jb b
        WHERE EXISTS (SELECT 1 FROM aoi_parcels p WHERE ST_Overlaps(p.geom, b.geom)))
        AS buildings_crossing
"""


@router.get("/{job_id}/validation")
def get_validation(job_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Summary + per-parcel GeoJSON. 409 until validation has been run."""
    job = _get_job_or_404(session, job_id)
    if job.validation_status is None:
        raise HTTPException(409, detail="validation not started for this job")

    minx, miny, maxx, maxy = job.bbox
    params = {"jid": str(job_id), "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}

    s = session.execute(text(_SUMMARY_SQL), params).one()
    summary = ValidationSummary(
        parcels_total=s.parcels_total,
        parcels_with_buildings=s.parcels_with_buildings,
        parcels_empty=s.parcels_total - s.parcels_with_buildings,
        buildings_total=s.buildings_total,
        buildings_off_parcel=s.buildings_off_parcel,
        buildings_crossing=s.buildings_crossing,
    )

    features = [
        {
            "type": "Feature",
            "geometry": json.loads(r.geometry),
            "properties": {
                "locator": r.locator,
                "address": r.address,
                "building_count": r.building_count,
                "crossing_count": r.crossing_count,
                "flag": (
                    "empty" if r.building_count == 0 else "multi" if r.building_count > 1 else "ok"
                ),
            },
        }
        for r in session.execute(text(_PARCELS_SQL), params).all()
    ]
    return {
        "status": job.validation_status,
        "error": job.validation_error,
        "summary": summary.model_dump(),
        "parcels": {"type": "FeatureCollection", "features": features},
    }
