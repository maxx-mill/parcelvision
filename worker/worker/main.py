"""Queue consumer loop. One RQ worker per container; scale with
`docker compose up --scale worker=N`."""

import logging

from redis import Redis
from rq import Queue, Worker

from .config import redis_url

QUEUE_NAME = "extraction"  # keep in sync with api/app/queue.py


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    conn = Redis.from_url(redis_url())
    worker = Worker([Queue(QUEUE_NAME, connection=conn)], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
