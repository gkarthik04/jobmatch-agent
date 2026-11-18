"""Single entry point for Gemini generation: retries, timeouts, token accounting, JSON caching."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.config import get_settings
from app.models import LLMCache
from app.services.embeddings import _retryable
from app.services.throttle import AsyncWindow

log = logging.getLogger("app.llm")


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class LLMResult:
    text: str = ""
    parsed: Any = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw_parts: Any = None  # provider content object, needed to continue tool-calling turns


class LLM(Protocol):
    async def generate(
        self,
        contents: Any,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult: ...


class GeminiLLM:
    def __init__(self):
        from google import genai

        s = get_settings()
        self.model, self.timeout = s.chat_model, s.llm_timeout_s
        self.client = genai.Client(api_key=s.gemini_api_key)
        self._window = AsyncWindow(s.llm_rpm)

    async def generate(
        self,
        contents: Any,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        from google.genai import types

        cfg: dict[str, Any] = {
            "temperature": temperature,
            "system_instruction": system,
            # Structured tasks don't need hidden reasoning tokens: cheaper and faster.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        if tools:
            cfg["tools"] = tools
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        config = types.GenerateContentConfig(**cfg)

        start = time.perf_counter()
        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(multiplier=2, max=40),
            stop=stop_after_attempt(5),
            retry=retry_if_exception(_retryable),
            reraise=True,
        ):
            with attempt:
                await self._window.acquire()
                res = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model, contents=contents, config=config
                    ),
                    self.timeout,
                )
        usage = res.usage_metadata
        out = LLMResult(
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
        cand = res.candidates[0] if res.candidates else None
        parts = (cand.content.parts if cand and cand.content and cand.content.parts else []) or []
        out.raw_parts = cand.content if cand else None
        out.text = "".join(p.text for p in parts if getattr(p, "text", None))
        out.function_calls = [
            FunctionCall(p.function_call.name, dict(p.function_call.args or {}))
            for p in parts
            if getattr(p, "function_call", None)
        ]
        if schema is not None:
            out.text = res.text or out.text
        log.info(
            "llm_call",
            extra={"model": self.model, "ms": round(out.latency_ms),
                   "prompt_tokens": out.prompt_tokens, "output_tokens": out.output_tokens},
        )
        return out


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = GeminiLLM()
    return _llm


def prompt_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


async def cached_json(session: AsyncSession, key: str, produce) -> dict:
    """Return cached JSON for `key`, otherwise await `produce()` (-> dict) and store it."""
    row = await session.get(LLMCache, key)
    if row is not None:
        return row.response
    value = await produce()
    session.add(LLMCache(key=key, response=value))
    await session.commit()
    return value
