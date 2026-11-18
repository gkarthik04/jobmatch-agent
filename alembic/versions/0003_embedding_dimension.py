"""resize embedding columns to settings.embed_dim (no-op if already that size)

Revision ID: 0003
Revises: 0002
"""
from alembic import op
from app.config import get_settings

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

DIM = get_settings().embed_dim

COLUMNS = [("jobs", "embedding"), ("embedding_cache", "embedding"), ("profiles", "pref_vector")]


def upgrade() -> None:
    # Existing vectors from another model/dimension are meaningless in the new space: drop them.
    for table, col in COLUMNS:
        op.execute(f"""
        DO $$
        DECLARE cur int;
        BEGIN
          SELECT atttypmod INTO cur FROM pg_attribute
           WHERE attrelid = '{table}'::regclass AND attname = '{col}';
          IF cur IS DISTINCT FROM {DIM} THEN
            {"DROP INDEX IF EXISTS ix_jobs_embedding_hnsw;" if table == "jobs" else ""}
            {"UPDATE jobs SET embedding_model = NULL;" if table == "jobs" else ""}
            {"DELETE FROM embedding_cache;" if table == "embedding_cache" else ""}
            ALTER TABLE {table} ALTER COLUMN {col} TYPE vector({DIM}) USING NULL;
          END IF;
        END $$;
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_jobs_embedding_hnsw ON jobs USING hnsw "
        "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    pass
