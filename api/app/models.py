"""Database models.

The worker writes to these same tables without importing this module (services
are built from separate contexts). If you change a table here, update the
matching SQL/column names in worker/worker/pipeline/load.py and
worker/worker/status.py.

Migrations: MVP creates tables with `Base.metadata.create_all` at API startup.
Alembic lands in Chapter 2 once the schema stops churning.
"""

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Job status contract — keep in sync with worker/worker/status.py
JOB_STATUSES = (
    "queued",
    "fetching_imagery",
    "running_inference",
    "vectorizing",
    "writing_db",
    "done",
    "failed",
    "canceled",
)
TERMINAL_STATUSES = {"done", "failed", "canceled"}


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    bbox: Mapped[list[float]] = mapped_column(JSON)
    backend: Mapped[str] = mapped_column(String(32), default="local_cpu")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    building_count: Mapped[int | None] = mapped_column(Integer, default=None)
    # Marks precomputed demo results loaded by `make demo` rather than live runs.
    is_seed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    # Regularization can emit Polygon or MultiPolygon; store generic geometry.
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="GEOMETRY", srid=4326))
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    area_sqm: Mapped[float | None] = mapped_column(Float, default=None)
