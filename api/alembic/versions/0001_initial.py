"""Initial schema: postgis extension, jobs, buildings.

Revision ID: 0001
Revises:
Create Date: 2026-07-23
"""

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("building_count", sa.Integer(), nullable=True),
        sa.Column("is_seed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_table(
        "buildings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id", sa.Uuid(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "geom",
            geoalchemy2.Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("area_sqm", sa.Float(), nullable=True),
    )
    op.create_index("ix_buildings_job_id", "buildings", ["job_id"])
    op.create_index("idx_buildings_geom", "buildings", ["geom"], postgresql_using="gist")


def downgrade() -> None:
    op.drop_table("buildings")
    op.drop_table("jobs")
