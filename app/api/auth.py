from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import AppError
from app.models import User
from app.security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)  # bcrypt ignores bytes past 72


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/signup", response_model=TokenOut, status_code=201)
async def signup(body: Credentials, session: AsyncSession = Depends(get_session)):
    email = body.email.lower()
    if (await session.execute(select(User.id).where(User.email == email))).first():
        raise AppError(409, "email_taken", "An account with this email already exists")
    user = User(email=email, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:  # race between two signups
        await session.rollback()
        raise AppError(409, "email_taken", "An account with this email already exists") from exc
    return TokenOut(access_token=create_token(user.id))


@router.post("/login", response_model=TokenOut)
async def login(body: Credentials, session: AsyncSession = Depends(get_session)):
    user = (
        await session.execute(select(User).where(User.email == body.email.lower()))
    ).scalar_one_or_none()
    # same message for unknown email and wrong password: no account enumeration
    if user is None or not verify_password(body.password, user.password_hash):
        raise AppError(401, "bad_credentials", "Incorrect email or password")
    return TokenOut(access_token=create_token(user.id))


@router.get("/me")
async def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email}
