"""Sliding-window rate limiter for LLM-backed endpoints.

In-memory and per-process: correct for a single Render instance (our deployment). With several
instances you would move the counters to Redis / Postgres. Keyed by user id, else client IP.
"""

import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import get_settings
from app.errors import AppError
from app.security import bearer_token, decode_token


class SlidingWindow:
    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> float:
        """Record a hit. Returns 0 if allowed, else seconds until the next slot frees up."""
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return self.window - (now - q[0])
        q.append(now)
        return 0.0

    def reset(self) -> None:
        self.hits.clear()


_limiter: SlidingWindow | None = None


def get_limiter() -> SlidingWindow:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindow(get_settings().rate_limit_per_minute)
    return _limiter


async def rate_limit(request: Request) -> None:
    """Dependency: 429 with Retry-After when the caller exceeds the per-minute budget."""
    token = bearer_token(request)
    try:
        key = f"u:{decode_token(token)}" if token else f"ip:{request.client.host if request.client else '?'}"
    except AppError:
        key = f"ip:{request.client.host if request.client else '?'}"
    retry = get_limiter().check(key)
    if retry > 0:
        raise AppError(429, "rate_limited", "Too many requests, slow down",
                       headers={"Retry-After": str(int(retry) + 1)})
