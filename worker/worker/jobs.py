"""RQ entrypoint. The API enqueues `worker.jobs.run_extraction` by dotted path
(api/app/queue.py) — renaming this function is an API change."""

import logging
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import text

from . import config, status
from .backends import get_backend
from .db import get_engine
from .pipeline.fetch import fetch_imagery
from .pipeline.load import load_buildings
from .pipeline.parcels import fetch_parcels, upsert_parcels, validate_job
from .pipeline.postprocess import postprocess

logger = logging.getLogger(__name__)


def run_extraction(job_id: str) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT bbox, backend FROM jobs WHERE id = :id"), {"id": job_id}
        ).one_or_none()
    if row is None:
        raise RuntimeError(f"job {job_id} not found in DB")
    bbox: list[float] = row.bbox
    backend = get_backend(row.backend or config.inference_backend())

    workdir = Path(tempfile.mkdtemp(prefix=f"job_{job_id[:8]}_", dir=_ensure_imagery_dir()))
    try:
        rasters: list[Path] = []
        if backend.needs_imagery:
            status.set_status(engine, job_id, status.FETCHING_IMAGERY)
            rasters = fetch_imagery(bbox, workdir, year=config.naip_year())
            logger.info("job %s: fetched %d NAIP tile(s)", job_id, len(rasters))

        status.set_status(engine, job_id, status.RUNNING_INFERENCE)
        raw = backend.extract(rasters, bbox)
        logger.info("job %s: %d raw detections (%s)", job_id, len(raw), backend.name)

        status.set_status(engine, job_id, status.VECTORIZING)
        clean = postprocess(raw, bbox)

        status.set_status(engine, job_id, status.WRITING_DB)
        count = load_buildings(engine, job_id, clean)

        status.set_status(engine, job_id, status.DONE, building_count=count)
        logger.info("job %s: done, %d buildings", job_id, count)
        return {"job_id": job_id, "building_count": count}
    except Exception as exc:
        status.set_status(engine, job_id, status.FAILED, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_validation(job_id: str) -> dict:
    """Chapter 3: validate a done job's footprints against county parcels.
    Enqueued by the API as `worker.jobs.run_validation`."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT bbox, status FROM jobs WHERE id = :id"), {"id": job_id}
        ).one_or_none()
    if row is None:
        raise RuntimeError(f"job {job_id} not found in DB")
    if row.status != status.DONE:
        raise RuntimeError(f"job {job_id} is {row.status}, not done — nothing to validate")
    bbox: list[float] = row.bbox

    try:
        status.set_validation_status(engine, job_id, status.LOADING_PARCELS)
        parcels = fetch_parcels(bbox)
        loaded = upsert_parcels(engine, parcels)
        logger.info("job %s: %d parcels in AOI (%d new)", job_id, len(parcels), loaded)

        status.set_validation_status(engine, job_id, status.VALIDATING)
        summary = validate_job(engine, job_id, bbox)

        status.set_validation_status(engine, job_id, status.DONE)
        return {"job_id": job_id, **summary}
    except Exception as exc:
        status.set_validation_status(
            engine, job_id, status.FAILED, error=f"{type(exc).__name__}: {exc}"
        )
        raise


def _ensure_imagery_dir() -> str:
    d = Path(config.imagery_dir())
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
