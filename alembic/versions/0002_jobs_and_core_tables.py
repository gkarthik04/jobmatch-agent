"""jobs index + core tables

Revision ID: 0002
Revises: 0001
"""
from alembic import op
from app.config import get_settings

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

DIM = get_settings().embed_dim


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE jobs (
            id SERIAL PRIMARY KEY,
            source_id VARCHAR(64) NOT NULL UNIQUE,
            title VARCHAR(500) NOT NULL,
            company VARCHAR(300),
            location VARCHAR(300),
            remote BOOLEAN NOT NULL DEFAULT FALSE,
            description TEXT NOT NULL,
            skills VARCHAR[] NOT NULL DEFAULT '{{}}',
            experience_level VARCHAR(100),
            posted_at TIMESTAMPTZ,
            content_hash VARCHAR(64) NOT NULL,
            embedding vector({DIM}),
            embedding_model VARCHAR(100),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(company, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(description, '')), 'B')
            ) STORED
        )
        """
    )
    op.execute("CREATE INDEX ix_jobs_tsv ON jobs USING gin (tsv)")
    op.execute(
        "CREATE INDEX ix_jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX ix_jobs_posted_at ON jobs (posted_at DESC)")
    op.execute("CREATE INDEX ix_jobs_remote ON jobs (remote)")

    op.execute(
        f"""CREATE TABLE embedding_cache (
            key VARCHAR(64) PRIMARY KEY, embedding vector({DIM}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute(
        """CREATE TABLE llm_cache (
            key VARCHAR(64) PRIMARY KEY, response JSON NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute(
        """CREATE TABLE users (
            id SERIAL PRIMARY KEY, email VARCHAR(320) NOT NULL UNIQUE,
            password_hash VARCHAR(200) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute(
        f"""CREATE TABLE profiles (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            data JSON NOT NULL, resume_text TEXT NOT NULL DEFAULT '',
            pref_vector vector({DIM}),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute(
        """CREATE TABLE feedback (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            liked BOOLEAN NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, job_id))"""
    )
    op.execute("CREATE INDEX ix_feedback_user ON feedback (user_id)")
    op.execute(
        """CREATE TABLE saved_jobs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, job_id))"""
    )
    op.execute("CREATE INDEX ix_saved_user ON saved_jobs (user_id)")
    op.execute(
        """CREATE TABLE messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL, content TEXT NOT NULL, meta JSON,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute("CREATE INDEX ix_messages_user ON messages (user_id, id)")
    op.execute(
        """CREATE TABLE agent_steps (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            run_id VARCHAR(32) NOT NULL, step INTEGER NOT NULL, kind VARCHAR(20) NOT NULL,
            name VARCHAR(100), args JSON, ok BOOLEAN NOT NULL DEFAULT TRUE, error TEXT,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    op.execute("CREATE INDEX ix_agent_steps_user ON agent_steps (user_id)")
    op.execute("CREATE INDEX ix_agent_steps_run ON agent_steps (run_id)")


def downgrade() -> None:
    for t in ("agent_steps", "messages", "saved_jobs", "feedback", "profiles", "users",
              "llm_cache", "embedding_cache", "jobs"):
        op.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
