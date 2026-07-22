import redis
from rq import Queue

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
    q.enqueue(EXTRACTION_FUNC, job_id, job_timeout=JOB_TIMEOUT_S, job_id=job_id)
