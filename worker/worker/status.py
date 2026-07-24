"""Job status contract — keep in sync with api/app/models.py."""

from sqlalchemy import Engine, text

QUEUED = "queued"
FETCHING_IMAGERY = "fetching_imagery"
RUNNING_INFERENCE = "running_inference"
VECTORIZING = "vectorizing"
WRITING_DB = "writing_db"
DONE = "done"
FAILED = "failed"

# Parcel-validation status (separate column, own lifecycle)
LOADING_PARCELS = "loading_parcels"
VALIDATING = "validating"


def set_status(
    engine: Engine,
    job_id: str,
    status: str,
    error: str | None = None,
    building_count: int | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status = :status, error = :error, "
                "building_count = COALESCE(:count, building_count), updated_at = now() "
                "WHERE id = :job_id"
            ),
            {"status": status, "error": error, "count": building_count, "job_id": job_id},
        )


def set_validation_status(
    engine: Engine, job_id: str, status: str, error: str | None = None
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET validation_status = :status, validation_error = :error, "
                "updated_at = now() WHERE id = :job_id"
            ),
            {"status": status, "error": error, "job_id": job_id},
        )
