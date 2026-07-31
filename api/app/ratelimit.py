"""A small Redis fixed-window rate limiter, used as a FastAPI dependency.

Job creation kicks off minutes of CPU inference, so an unbounded POST /jobs is a
trivial denial-of-service. This caps creates per client IP per minute using a
single INCR + EXPIRE against the Redis we already run — so it holds across API
replicas, unlike an in-process counter. Redis being down fails open (availability
over strictness); the AOI-size cap is the harder backstop.
"""

import logging

from fastapi import HTTPException, Request

from .config import get_settings
from .queue import get_redis

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    # nginx forwards the real client in X-Real-IP / X-Forwarded-For; fall back to
    # the socket peer for direct connections.
    fwd = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_jobs(request: Request) -> None:
    """Dependency: allow up to job_rate_limit_per_min creates per IP per minute."""
    limit = get_settings().job_rate_limit_per_min
    if limit <= 0:
        return
    ip = _client_ip(request)
    key = f"ratelimit:jobs:{ip}"
    try:
        redis = get_redis()
        count = redis.incr(key)
        if count == 1:
            redis.expire(key, 60)
    except Exception:  # noqa: BLE001 — availability over strictness
        logger.warning("rate limiter unavailable; allowing request")
        return
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({limit} jobs/min); slow down",
            headers={"Retry-After": "60"},
        )
