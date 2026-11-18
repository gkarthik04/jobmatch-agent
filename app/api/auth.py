from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import AppError
from app.models import User
from app.security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def password_problem(password: str, email: str = "") -> str | None:
    """Signup password policy. The browser shows the same rules live; the server is the authority."""
    if len(password) < 8:
        return "Use at least 8 characters."
    if len(password) > 72:  # bcrypt ignores bytes past 72
        return "Use at most 72 characters."
    if not (any(c.islower() for c in password) and any(c.isupper() for c in password)):
        return "Use both upper case and lower case letters."
    if not any(c.isdigit() for c in password):
        return "Add at least one number."
    local = email.split("@")[0].lower()
    if len(local) >= 4 and local in password.lower():
        return "Do not include your email name in the password."
    return None


class LoginIn(BaseModel):
    """Login does not apply the signup policy, so accounts made under older rules keep working."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class SignupIn(LoginIn):
    @model_validator(mode="after")
    def _check_password(self):
        problem = password_problem(self.password, str(self.email))
        if problem:
            raise ValueError(problem)
        return self


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/signup", response_model=TokenOut, status_code=201)
async def signup(body: SignupIn, session: AsyncSession = Depends(get_session)):
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
async def login(body: LoginIn, session: AsyncSession = Depends(get_session)):
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
