import redis
from rq import Queue, Retry

from .config import get_settings

QUEUE_NAME = "extraction"

# Referenced by dotted path so the API image never imports (or installs) the
# worker's ML stack. Keep in sync with worker/worker/jobs.py.
EXTRACTION_FUNC = "worker.jobs.run_extraction"

# CPU inference on a capped AOI runs minutes, not hours; this is the hard stop.
JOB_TIMEOUT_S = 3600


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url)


def enqueue_extraction(job_id: str) -> None:
    q = Queue(QUEUE_NAME, connection=get_redis())
    # Two retries with backoff ride out transient upstream faults (Planetary
    # Computer 504s, redis blips). Reruns are safe: the worker clears any
    # partial buildings for the job before writing.
    q.enqueue(
        EXTRACTION_FUNC,
        job_id,
        job_timeout=JOB_TIMEOUT_S,
        job_id=job_id,
        retry=Retry(max=2, interval=[30, 120]),
    )


def cancel_extraction(job_id: str, running: bool) -> None:
    """Best-effort RQ-side cancellation; the DB row is the source of truth.
    Queued jobs are pulled from the queue; running ones get their work horse
    stopped (the DB status the API writes afterwards survives the kill)."""
    import rq.command
    import rq.exceptions
    import rq.job

    conn = get_redis()
    try:
        if running:
            rq.command.send_stop_job_command(conn, job_id)
        else:
            rq.job.Job.fetch(job_id, connection=conn).cancel()
    except (rq.exceptions.NoSuchJobError, rq.exceptions.InvalidJobOperation):
        pass  # already gone or finished between our check and the command
