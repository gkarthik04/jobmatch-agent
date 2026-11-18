"""title-only tsvector + GIN index (third retriever: title-field match)

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS title_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, ''))) STORED"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_jobs_title_tsv ON jobs USING gin (title_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_jobs_title_tsv")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS title_tsv")
