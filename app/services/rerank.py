"""LLM reranker: one Gemini call scores the fused top-N for relevance to the query.

Why LLM and not a cross-encoder: a local cross-encoder needs torch (~1GB image) which does not
fit Render's free 512MB RAM. Pointwise scores in a single JSON-schema call keep latency/cost low.
Results are cached by (query, candidate ids), and any failure falls back to the RRF order.
"""

import logging

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm import cached_json, get_llm, prompt_key
from app.services.search import Hit

log = logging.getLogger("app.rerank")

SYSTEM = (
    "You are a recruiter ranking job postings for a search query or candidate profile. "
    "Score each posting 0-10 for how well it matches: role/title fit first, then required skills, "
    "seniority and location/remote constraints. Be discriminating: use the full range."
)


class JobScore(BaseModel):
    id: int
    score: float = Field(ge=0, le=10)


class RerankOut(BaseModel):
    scores: list[JobScore]


def _render(hit: Hit) -> str:
    j = hit.job
    skills = ", ".join((j.skills or [])[:10])
    snippet = (j.description or "")[:350].replace("\n", " ")
    return f"[{j.id}] {j.title} | {j.company} | {j.location}{' (remote)' if j.remote else ''} | " \
           f"skills: {skills} | {snippet}"


def apply_scores(hits: list[Hit], scores: dict[int, float]) -> list[Hit]:
    """Sort by rerank score desc; ties and unscored items keep their fused (RRF) order."""
    for h in hits:
        h.rerank_score = scores.get(h.job.id)
    order = {h.job.id: i for i, h in enumerate(hits)}
    return sorted(
        hits, key=lambda h: (-(h.rerank_score if h.rerank_score is not None else -1), order[h.job.id])
    )


async def rerank(session: AsyncSession, llm, query: str, hits: list[Hit]) -> list[Hit]:
    if len(hits) < 2:
        return hits
    llm = llm or get_llm()
    key = prompt_key("rerank-v1", query, [h.job.id for h in hits])

    async def produce() -> dict:
        prompt = f"Query / profile:\n{query[:1500]}\n\nPostings:\n" + "\n".join(_render(h) for h in hits)
        res = await llm.generate(prompt, system=SYSTEM, schema=RerankOut)
        return RerankOut.model_validate_json(res.text).model_dump()

    try:
        data = await cached_json(session, key, produce)
        scores = {s["id"]: s["score"] for s in data["scores"]}
    except Exception:  # noqa: BLE001 - reranking is an enhancement; never fail the search
        log.warning("rerank failed, falling back to RRF order", exc_info=True)
        return hits
    return apply_scores(hits, scores)
