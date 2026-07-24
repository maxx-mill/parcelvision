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


# Per-parcel validation, computed live from the same spatial joins the worker
# summarized. A building is assigned to the parcel containing its interior point;
# a footprint "crosses" when it geometrically overlaps a parcel boundary.
_PARCEL_SQL = """
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


@router.get("/{job_id}/validation")
def get_validation(job_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Summary + per-parcel GeoJSON. 409 until validation has been run."""
    job = _get_job_or_404(session, job_id)
    if job.validation_status is None:
        raise HTTPException(409, detail="validation not started for this job")

    minx, miny, maxx, maxy = job.bbox
    params = {"jid": str(job_id), "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
    rows = session.execute(text(_PARCEL_SQL), params).all()

    features = []
    parcels_with = buildings_on = crossing = 0
    for r in rows:
        if r.building_count > 0:
            parcels_with += 1
            buildings_on += r.building_count
        crossing += r.crossing_count
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(r.geometry),
                "properties": {
                    "locator": r.locator,
                    "address": r.address,
                    "building_count": r.building_count,
                    "crossing_count": r.crossing_count,
                    # UI shading class
                    "flag": (
                        "empty"
                        if r.building_count == 0
                        else "multi"
                        if r.building_count > 1
                        else "ok"
                    ),
                },
            }
        )

    total_buildings = session.execute(
        text("SELECT count(*) FROM buildings WHERE job_id = :jid"), {"jid": str(job_id)}
    ).scalar_one()
    summary = ValidationSummary(
        parcels_total=len(rows),
        parcels_with_buildings=parcels_with,
        parcels_empty=len(rows) - parcels_with,
        buildings_total=total_buildings,
        buildings_off_parcel=total_buildings - buildings_on,
        buildings_crossing=crossing,
    )
    return {
        "status": job.validation_status,
        "error": job.validation_error,
        "summary": summary.model_dump(),
        "parcels": {"type": "FeatureCollection", "features": features},
    }
