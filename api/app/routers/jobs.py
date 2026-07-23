import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_session
from ..models import TERMINAL_STATUSES, Building, Job
from ..queue import cancel_extraction, enqueue_extraction
from ..schemas import JobCreate, JobOut, bbox_area_km2

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobOut, status_code=201)
def create_job(
    payload: JobCreate,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Job:
    area = bbox_area_km2(payload.bbox)
    if area > settings.aoi_bbox_limit_km2:
        raise HTTPException(
            status_code=413,
            detail=(
                f"bbox is {area:.2f} km²; limit is {settings.aoi_bbox_limit_km2} km². "
                "Draw a smaller area — CPU inference on large AOIs would run for hours."
            ),
        )
    # Explicit values (not column defaults) so the id exists before enqueue.
    job = Job(
        id=uuid.uuid4(),
        status="queued",
        bbox=payload.bbox,
        backend=settings.inference_backend,
        is_seed=False,
        created_at=datetime.now(UTC),
    )
    session.add(job)
    session.commit()
    enqueue_extraction(str(job.id))
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = 20, session: Session = Depends(get_session)) -> list[Job]:
    rows = session.execute(
        select(Job).order_by(Job.created_at.desc()).limit(min(limit, 100))
    ).scalars()
    return list(rows)


def _get_job_or_404(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, session: Session = Depends(get_session)) -> Job:
    return _get_job_or_404(session, job_id)


@router.delete("/{job_id}", response_model=JobOut)
def cancel_job(job_id: uuid.UUID, session: Session = Depends(get_session)) -> Job:
    """Cancel a queued or running job. Terminal jobs are left untouched (409)."""
    job = _get_job_or_404(session, job_id)
    if job.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail=f"job already {job.status}")
    cancel_extraction(str(job_id), running=job.status != "queued")
    job.status = "canceled"
    job.error = None
    session.commit()
    return job


@router.get("/{job_id}/buildings")
def get_buildings(job_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Detected footprints as a GeoJSON FeatureCollection (EPSG:4326)."""
    _get_job_or_404(session, job_id)
    rows = session.execute(
        select(
            Building.id,
            func.ST_AsGeoJSON(Building.geom).label("geometry"),
            Building.confidence,
            Building.area_sqm,
        ).where(Building.job_id == job_id)
    ).all()
    features = [
        {
            "type": "Feature",
            "id": r.id,
            "geometry": json.loads(r.geometry),
            "properties": {"confidence": r.confidence, "area_sqm": r.area_sqm},
        }
        for r in rows
    ]
    return {"type": "FeatureCollection", "features": features}
