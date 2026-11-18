"""Single entry point for Gemini generation: retries, timeouts, token accounting, JSON caching."""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import LLMCache
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
    model: str | None = None


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


class LLMUnavailable(Exception):
    """Every model in the chain is rate-limited, over daily quota, or down."""


def classify_error(exc: BaseException) -> tuple[str, float]:
    """-> (kind, retry_after_s). kinds: daily | minute | down | unsupported | bad_request | other."""
    msg = str(exc)
    low = msg.lower()
    code = getattr(exc, "code", None)
    delay = 30.0
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg) or re.search(
        r"retryDelay': '(?:(\d+)h)?(?:(\d+)m)?(\d+)s'", msg)
    if m:
        delay = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    if code == 429 or "resource_exhausted" in low:
        if "per day" in low or "perday" in low:
            return "daily", max(delay, 3600.0)
        return "minute", min(max(delay, 5.0), 120.0)
    if code in (500, 502, 503, 504) or "unavailable" in low or "timeout" in low:
        return "down", 60.0
    if code == 404:
        return "unsupported", 24 * 3600.0
    if code == 400:
        return "bad_request", 0.0
    return "other", 0.0


class GeminiLLM:
    """Gemini via google-genai with an ordered model chain.

    Free-tier quotas are per model (and tiny: e.g. 20 requests/day on one Flash model), so a call
    walks the chain: a model that returns a daily-quota 429 is parked until it resets, a
    per-minute 429 / 5xx parks it briefly, and the next model serves the request. A proactive
    sliding-window limiter keeps us under per-minute limits in the first place.
    """

    def __init__(self, models: list[str] | None = None):
        from google import genai
        from google.genai import types

        s = get_settings()
        self.models = models or s.chat_model_chain
        self.timeout = s.llm_timeout_s
        self.model = self.models[0]
        # system_instruction / function calling need v1beta; the SDK default can be v1
        self.client = genai.Client(
            api_key=s.gemini_api_key, http_options=types.HttpOptions(api_version="v1beta")
        )
        self._window = AsyncWindow(s.llm_rpm)
        self._parked: dict[str, float] = {}  # model -> monotonic time it may be tried again
        self._no_thinking_cfg: set[str] = set()  # models that reject thinking_budget=0
        self.last_model: str | None = None

    def _config(self, model, system, schema, tools, temperature):
        from google.genai import types

        cfg: dict[str, Any] = {"temperature": temperature, "system_instruction": system}
        if model not in self._no_thinking_cfg:
            # Structured tasks don't need hidden reasoning tokens: cheaper and faster.
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        if tools:
            cfg["tools"] = tools
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        return types.GenerateContentConfig(**cfg)

    async def _call(self, model, contents, system, schema, tools, temperature):
        return await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=model, contents=contents,
                config=self._config(model, system, schema, tools, temperature),
            ),
            self.timeout,
        )

    async def generate(
        self,
        contents: Any,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        start = time.perf_counter()
        last_exc: BaseException | None = None
        for attempt_pass in range(2):
            tried = False
            for model in self.models:
                if self._parked.get(model, 0) > time.monotonic():
                    continue
                tried = True
                await self._window.acquire()
                try:
                    try:
                        res = await self._call(model, contents, system, schema, tools, temperature)
                    except Exception as exc:  # noqa: BLE001
                        if (classify_error(exc)[0] == "bad_request"
                                and model not in self._no_thinking_cfg):
                            # some models reject thinking_budget=0: retry once without the knob
                            self._no_thinking_cfg.add(model)
                            res = await self._call(model, contents, system, schema, tools, temperature)
                        else:
                            raise
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    kind, delay = classify_error(exc)
                    log.warning("llm model failed", extra={"model": model, "kind": kind,
                                                           "retry_after": round(delay),
                                                           "error": str(exc)[:200]})
                    # bad_request after the retry above = this model can't serve this request type
                    self._parked[model] = time.monotonic() + (delay or 1800)
                    continue
                return self._to_result(res, model, schema, start)
            if not tried:  # everything parked: wait briefly if a per-minute limit frees up soon
                soonest = min(self._parked.values(), default=0) - time.monotonic()
                if soonest > 20 or attempt_pass == 1:
                    break
                await asyncio.sleep(max(soonest, 1))
        raise LLMUnavailable(f"all models unavailable: {last_exc}")

    def _to_result(self, res, model, schema, start) -> LLMResult:
        usage = res.usage_metadata
        out = LLMResult(
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=model,
        )
        self.last_model = model
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
            extra={"model": model, "ms": round(out.latency_ms),
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
