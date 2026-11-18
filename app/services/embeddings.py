"""Gemini embeddings with batching, retry/backoff, normalisation and a DB-backed cache."""

import asyncio
import hashlib
import logging
import math
import time
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.config import get_settings
from app.models import EmbeddingCache

log = logging.getLogger("app.embeddings")

DOC = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


class Embedder(Protocol):
    async def embed(self, texts: list[str], task: str = DOC) -> list[list[float]]: ...


def normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "code", None)
    return code in (429, 500, 502, 503, 504) or "429" in msg or "resource_exhausted" in msg \
        or "unavailable" in msg or "timeout" in msg


class GeminiEmbedder:
    """Batches texts, retries 429/5xx with jittered exponential backoff (free-tier friendly)."""

    def __init__(self, batch_size: int = 40, items_per_minute: int = 90):
        from google import genai

        s = get_settings()
        self.model, self.dim = s.embed_model, s.embed_dim
        self.client = genai.Client(api_key=s.gemini_api_key)
        self.batch_size, self.items_per_minute = batch_size, items_per_minute
        self._sent: list[tuple[float, int]] = []  # (monotonic time, n items) in the last minute

    async def _throttle(self, n: int) -> None:
        """Free tier counts every text in a batch as a request: 100/min. Stay under with a window."""
        while True:
            now = time.monotonic()
            self._sent = [(t, k) for t, k in self._sent if now - t < 60]
            if sum(k for _, k in self._sent) + n <= self.items_per_minute or not self._sent:
                self._sent.append((now, n))
                return
            await asyncio.sleep(max(0.5, 60 - (now - self._sent[0][0]) + 0.1))

    async def _one_batch(self, batch: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(multiplier=2, max=60),
            stop=stop_after_attempt(7),
            retry=retry_if_exception(_retryable),
            reraise=True,
        ):
            with attempt:
                await self._throttle(len(batch))
                res = await self.client.aio.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dim, task_type=task
                    ),
                )
                return [normalize(list(e.values)) for e in res.embeddings]
        raise RuntimeError("unreachable")

    async def embed(self, texts: list[str], task: str = DOC) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(await self._one_batch(texts[i : i + self.batch_size], task))
        return out


def cache_key(model: str, task: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{task}|{text}".encode()).hexdigest()


async def embed_cached(
    session: AsyncSession, embedder: Embedder, texts: list[str], task: str = QUERY
) -> list[list[float]]:
    """Embed `texts`, reading/writing the embedding_cache table (used for queries)."""
    model = get_settings().embed_model
    keys = [cache_key(model, task, t) for t in texts]
    rows = (
        await session.execute(select(EmbeddingCache).where(EmbeddingCache.key.in_(keys)))
    ).scalars()
    found = {r.key: [float(x) for x in r.embedding] for r in rows}
    missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in found]
    if missing:
        uniq = list({k: t for k, t in missing}.items())
        vecs = await embedder.embed([t for _, t in uniq], task)
        for (k, _), v in zip(uniq, vecs, strict=True):
            found[k] = v
            await session.merge(EmbeddingCache(key=k, embedding=v))
        await session.commit()
    return [found[k] for k in keys]
