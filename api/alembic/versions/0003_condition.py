"""Chapter 6: per-structure roof condition indicators.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("buildings", sa.Column("condition", sa.String(16), nullable=True))
    op.add_column("buildings", sa.Column("tarp_fraction", sa.Float(), nullable=True))
    op.add_column("buildings", sa.Column("heterogeneity", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("buildings", "heterogeneity")
    op.drop_column("buildings", "tarp_fraction")
    op.drop_column("buildings", "condition")
