"""Persist the meal selected by the app."""
import sqlalchemy as sa
from alembic import op

revision = "0002_meal_slot"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meal_sessions", sa.Column("meal", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("meal_sessions", "meal")
