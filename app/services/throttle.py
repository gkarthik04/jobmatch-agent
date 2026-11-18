"""Async sliding-window throttle for provider rate limits (free-tier friendly)."""

import asyncio
import time


class AsyncWindow:
    """Allow at most `limit` units per `window_s`. `acquire(n)` sleeps until n units fit."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self._sent: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._sent = [(t, k) for t, k in self._sent if now - t < self.window]
                if not self._sent or sum(k for _, k in self._sent) + n <= self.limit:
                    self._sent.append((now, n))
                    return
                await asyncio.sleep(max(0.2, self.window - (now - self._sent[0][0]) + 0.05))
