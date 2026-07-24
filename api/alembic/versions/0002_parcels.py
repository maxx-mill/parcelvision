"""Chapter 3: parcels table + per-job validation status columns.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-24
"""

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("validation_status", sa.String(32), nullable=True))
    op.add_column("jobs", sa.Column("validation_error", sa.Text(), nullable=True))

    op.create_table(
        "parcels",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("locator", sa.String(64), nullable=False),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column(
            "geom",
            geoalchemy2.Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
        ),
        sa.Column("loaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_parcels_locator", "parcels", ["locator"], unique=True)
    op.create_index("idx_parcels_geom", "parcels", ["geom"], postgresql_using="gist")


def downgrade() -> None:
    op.drop_table("parcels")
    op.drop_column("jobs", "validation_error")
    op.drop_column("jobs", "validation_status")
