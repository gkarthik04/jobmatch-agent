from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import AppError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness: no dependencies, always cheap."""
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(session: AsyncSession = Depends(get_session)):
    """Readiness: verifies the database is reachable."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise AppError(503, "db_unavailable", "Database not reachable") from exc
    return {"status": "ready"}
