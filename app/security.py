"""Password hashing (bcrypt), JWT access tokens, and the current-user dependency."""

import time

import bcrypt
import jwt
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.errors import AppError
from app.models import User

ALGO = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(user_id: int, now: float | None = None) -> str:
    s = get_settings()
    iat = int(now or time.time())
    payload = {"sub": str(user_id), "iat": iat, "exp": iat + s.jwt_expire_minutes * 60}
    return jwt.encode(payload, s.jwt_secret, algorithm=ALGO)


def decode_token(token: str) -> int:
    try:
        data = jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGO])
        return int(data["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise AppError(401, "invalid_token", "Invalid or expired token",
                       headers={"WWW-Authenticate": "Bearer"}) from exc


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> User:
    token = bearer_token(request)
    if not token:
        raise AppError(401, "not_authenticated", "Missing bearer token",
                       headers={"WWW-Authenticate": "Bearer"})
    user = await session.get(User, decode_token(token))
    if user is None:
        raise AppError(401, "invalid_token", "User no longer exists")
    return user
