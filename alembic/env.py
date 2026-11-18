import asyncio
import os

from alembic import context
from app import models  # noqa: F401  (register tables on Base.metadata)
from app.db import make_engine
from app.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    from app.config import get_settings

    return os.environ.get("ALEMBIC_DATABASE_URL") or get_settings().database_url


def _do_run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async() -> None:
    engine = make_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async())


if context.is_offline_mode():
    raise SystemExit("offline mode is not supported")
run_migrations_online()
