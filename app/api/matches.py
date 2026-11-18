from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import get_embedder, get_llm_dep
from app.errors import AppError
from app.models import Feedback, Job, Profile, SavedJob, User
from app.ratelimit import rate_limit
from app.security import current_user
from app.services.matching import MatchOut, get_matches
from app.services.preferences import preference_delta
from app.services.search import SearchFilters

router = APIRouter(tags=["matches"])


@router.get("/matches", response_model=list[MatchOut], dependencies=[Depends(rate_limit)])
async def matches(
    limit: int = Query(10, ge=1, le=20),
    remote: bool | None = None,
    location: str | None = None,
    personalise: bool = True,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    embedder=Depends(get_embedder),
    llm=Depends(get_llm_dep),
):
    """Top matches for the user's resume profile, personalised by past likes/dislikes."""
    if await session.get(Profile, user.id) is None:
        raise AppError(409, "no_profile", "Upload a resume first (POST /resume)")
    return await get_matches(
        session, embedder, llm, user.id, k=limit,
        filters=SearchFilters(remote=remote, location=location), personalise=personalise,
    )


class FeedbackIn(BaseModel):
    job_id: int
    liked: bool


async def update_preference(session: AsyncSession, user_id: int) -> int:
    """Recompute the user's Rocchio feedback vector from all their feedback rows."""
    rows = (
        await session.execute(
            select(Feedback.liked, Job.embedding)
            .join(Job, Job.id == Feedback.job_id)
            .where(Feedback.user_id == user_id, Job.embedding.is_not(None))
        )
    ).all()
    liked = [[float(x) for x in e] for ok, e in rows if ok]
    disliked = [[float(x) for x in e] for ok, e in rows if not ok]
    profile = await session.get(Profile, user_id)
    profile.pref_vector = preference_delta(liked, disliked)
    await session.commit()
    return len(rows)


@router.post("/feedback")
async def feedback(
    body: FeedbackIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    if await session.get(Profile, user.id) is None:
        raise AppError(409, "no_profile", "Upload a resume first (POST /resume)")
    if await session.get(Job, body.job_id) is None:
        raise AppError(404, "job_not_found", "No such job")
    existing = (
        await session.execute(
            select(Feedback).where(Feedback.user_id == user.id, Feedback.job_id == body.job_id)
        )
    ).scalar_one_or_none()
    if existing:
        existing.liked = body.liked
    else:
        session.add(Feedback(user_id=user.id, job_id=body.job_id, liked=body.liked))
    await session.commit()
    n = await update_preference(session, user.id)
    return {"ok": True, "feedback_count": n}


class SavedIn(BaseModel):
    job_id: int


@router.get("/saved")
async def saved(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    rows = (
        await session.execute(
            select(Job).join(SavedJob, SavedJob.job_id == Job.id)
            .where(SavedJob.user_id == user.id).order_by(SavedJob.id.desc())
        )
    ).scalars().all()
    from types import SimpleNamespace

    from app.schemas import job_out

    return [job_out(SimpleNamespace(job=j, score=0.0, ranks={}, rerank_score=None)) for j in rows]


@router.post("/saved", status_code=201)
async def save(
    body: SavedIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    if await session.get(Job, body.job_id) is None:
        raise AppError(404, "job_not_found", "No such job")
    await session.execute(
        pg_insert(SavedJob).values(user_id=user.id, job_id=body.job_id).on_conflict_do_nothing()
    )
    await session.commit()
    return {"ok": True}


@router.delete("/saved/{job_id}", status_code=204)
async def unsave(
    job_id: int,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import delete

    await session.execute(
        delete(SavedJob).where(SavedJob.user_id == user.id, SavedJob.job_id == job_id)
    )
    await session.commit()
