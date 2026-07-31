"""Request logging, correlation IDs, and uniform error handling for the API.

Every request gets an X-Request-ID (honoured if the caller sends one, else
generated), bound to a ContextVar so it appears in every log line for that
request — the thing you actually need when debugging a production trace. One
access line per request records method, path, status, and latency. Unhandled
exceptions are logged with a traceback and returned as a clean JSON 500 carrying
the request id, so internals never leak to the client.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

access_log = logging.getLogger("parcelvision.access")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Root logging with request-id-aware formatting. Idempotent."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s")
    )
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # We emit our own access line; silence uvicorn's to avoid duplicates.
    logging.getLogger("uvicorn.access").disabled = True


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = rid
        access_log.info(
            "%s %s -> %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        access_log.exception("%s %s -> 500 (%.0fms)", request.method, request.url.path, elapsed)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    finally:
        request_id_var.reset(token)
