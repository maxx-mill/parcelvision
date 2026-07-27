"""Chapter 6 damage v3: in-domain classifier roof_damage_score.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("buildings", sa.Column("roof_damage_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("buildings", "roof_damage_score")
