"""Queue consumer loop. One RQ worker per container; scale with
`docker compose up --scale worker=N`."""

import logging

from redis import Redis
from rq import Queue, Worker
from rq.registry import StartedJobRegistry
from sqlalchemy import text

from .config import redis_url
from .db import get_engine

QUEUE_NAME = "extraction"  # keep in sync with api/app/queue.py

logger = logging.getLogger(__name__)


def reconcile_orphans(conn: Redis) -> None:
    """A worker that dies mid-job (crash, redis timeout, forced recreate)
    leaves DB rows stuck in an active stage with no RQ job behind them. Fail
    them at boot so users see a resolvable state instead of a frozen spinner.
    The updated_at guard avoids racing a job submitted this instant."""
    live = set(Queue(QUEUE_NAME, connection=conn).job_ids)
    live |= set(StartedJobRegistry(QUEUE_NAME, connection=conn).get_job_ids())
    engine = get_engine()
    with engine.begin() as db:
        rows = db.execute(
            text(
                "UPDATE jobs SET status = 'failed', "
                "error = 'orphaned: worker restarted mid-job — resubmit', updated_at = now() "
                "WHERE status NOT IN ('done', 'failed', 'canceled') "
                "AND updated_at < now() - interval '30 seconds' "
                "AND NOT (id::text = ANY(:live)) RETURNING id"
            ),
            {"live": list(live)},
        ).all()
    for (job_id,) in rows:
        logger.warning("reconciled orphaned job %s -> failed", job_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    conn = Redis.from_url(redis_url())
    reconcile_orphans(conn)
    worker = Worker([Queue(QUEUE_NAME, connection=conn)], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
