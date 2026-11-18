from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

EMBED_DIM = get_settings().embed_dim


class Base(DeclarativeBase):
    pass


class Job(Base):
    """A job posting. `tsv` (generated tsvector) exists only in the DB, see migration 0002."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    company: Mapped[str | None] = mapped_column(String(300))
    location: Mapped[str | None] = mapped_column(String(300))
    remote: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text)
    skills: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    experience_level: Mapped[str | None] = mapped_column(String(100))
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding = mapped_column(Vector(EMBED_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingCache(Base):
    """Query/text embeddings cached by content hash so we never pay twice."""

    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(model|task|text)
    embedding = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LLMCache(Base):
    """Deterministic LLM results (rerank, resume parse, judge) cached by prompt hash."""

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Profile(Base):
    """Parsed resume + learned preference vector (one per user)."""

    __tablename__ = "profiles"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)  # validated ProfileSchema
    resume_text: Mapped[str] = mapped_column(Text, default="")
    pref_vector = mapped_column(Vector(EMBED_DIM), nullable=True)  # Rocchio-updated
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("user_id", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    liked: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SavedJob(Base):
    __tablename__ = "saved_jobs"
    __table_args__ = (UniqueConstraint("user_id", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    """Conversation memory: one row per turn item (user / assistant / tool)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSON)  # e.g. last result job ids for follow-ups
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    step: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))  # llm | tool
    name: Mapped[str | None] = mapped_column(String(100))
    args: Mapped[dict | None] = mapped_column(JSON)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
