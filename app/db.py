"""Async SQLAlchemy engine/session. Engine is created lazily so imports never need a DB."""

from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def normalize_url(url: str) -> tuple[str, dict]:
    """Turn a Neon/libpq style URL into an asyncpg SQLAlchemy URL + connect_args.

    Neon gives ``postgresql://...?sslmode=require&channel_binding=require``; asyncpg wants
    ``postgresql+asyncpg://`` and ``ssl=...`` instead, and rejects unknown query params.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    query = dict(parse_qsl(parts.query))
    connect_args: dict = {}
    sslmode = query.pop("sslmode", None)
    ssl = query.pop("ssl", None)
    query.pop("channel_binding", None)
    if sslmode in ("require", "verify-ca", "verify-full") or ssl in ("require", "true", "1"):
        connect_args["ssl"] = "require"
    clean = urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return clean, connect_args


def make_engine(url: str | None = None) -> AsyncEngine:
    clean, connect_args = normalize_url(url or get_settings().database_url)
    return create_async_engine(
        clean,
        connect_args=connect_args,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,  # Neon suspends idle computes; drop dead connections
        pool_recycle=300,
    )


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = make_engine()
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
