"""Job status contract — keep in sync with api/app/models.py."""

from sqlalchemy import Engine, text

QUEUED = "queued"
FETCHING_IMAGERY = "fetching_imagery"
RUNNING_INFERENCE = "running_inference"
VECTORIZING = "vectorizing"
WRITING_DB = "writing_db"
DONE = "done"
FAILED = "failed"


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
