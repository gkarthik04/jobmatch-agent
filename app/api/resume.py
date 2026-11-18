from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import get_llm_dep
from app.errors import AppError
from app.models import Profile, User
from app.ratelimit import rate_limit
from app.schemas import ProfileSchema
from app.security import current_user
from app.services.resume import MAX_BYTES, extract_pdf_text, parse_resume

router = APIRouter(tags=["resume"])


@router.post("/resume", response_model=ProfileSchema, dependencies=[Depends(rate_limit)])
async def upload_resume(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    llm=Depends(get_llm_dep),
):
    """Upload a PDF resume; returns (and stores) the validated structured profile."""
    data = await file.read(MAX_BYTES + 1)
    text = extract_pdf_text(data)
    profile = await parse_resume(llm, text)
    row = await session.get(Profile, user.id)
    if row is None:
        session.add(Profile(user_id=user.id, data=profile.model_dump(), resume_text=text))
    else:
        row.data, row.resume_text, row.pref_vector = profile.model_dump(), text, None
    await session.commit()
    return profile


@router.get("/profile", response_model=ProfileSchema)
async def get_profile(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    row = await session.get(Profile, user.id)
    if row is None:
        raise AppError(404, "no_profile", "Upload a resume first")
    return ProfileSchema.model_validate(row.data)
