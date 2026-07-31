import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .observability import configure_logging, request_context_middleware
from .routers import exports, health, jobs, parcels, validation

configure_logging(get_settings().log_level)
logger = logging.getLogger("parcelvision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "starting ParcelVision API (env=%s, backend=%s)",
        settings.app_env,
        settings.inference_backend,
    )
    init_db()
    yield


app = FastAPI(title="ParcelVision API", version="0.1.0", lifespan=lifespan)

# Outermost middleware: request id + access log + uniform 500s wrap everything.
app.middleware("http")(request_context_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# nginx (frontend container) proxies /api/* here, so all routes live under /api.
app.include_router(health.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(validation.router, prefix="/api")
app.include_router(parcels.router, prefix="/api")
app.include_router(exports.router, prefix="/api")
