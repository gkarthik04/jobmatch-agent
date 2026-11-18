"""apply links, source, expiry and live-feed freshness on jobs

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS url TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS apply_url TEXT")
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS source VARCHAR(30) NOT NULL DEFAULT 'linkedin_2024'"
    )
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ")
    op.execute("CREATE INDEX IF NOT EXISTS ix_jobs_source_seen ON jobs (source, last_seen_at DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_jobs_source_seen")
    for col in ("last_seen_at", "expires_at", "source", "apply_url", "url"):
        op.execute(f"ALTER TABLE jobs DROP COLUMN IF EXISTS {col}")
