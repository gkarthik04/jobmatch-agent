"""Profile -> ranked matches with structured, grounded explanations."""

import asyncio
import logging

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Feedback, Job, Profile
from app.schemas import JobOut, ProfileSchema, job_out
from app.services.embeddings import QUERY, embed_cached
from app.services.llm import cached_json, get_llm, prompt_key
from app.services.preferences import apply_preference
from app.services.search import Hit, SearchFilters, search
from app.services.skills import extract_skills

log = logging.getLogger("app.matching")


class MatchExplanation(BaseModel):
    job_id: int
    why_it_matches: str
    fit_score: int = Field(ge=0, le=100)


class ExplainOut(BaseModel):
    items: list[MatchExplanation]


class MatchOut(BaseModel):
    job: JobOut
    why_it_matches: str
    matched_skills: list[str]
    missing_skills: list[str]
    fit_score: int


def profile_query_text(p: ProfileSchema) -> str:
    parts = []
    if p.preferred_roles:
        parts.append("Target roles: " + ", ".join(p.preferred_roles))
    parts.append("Skills: " + ", ".join(p.skills[:30]))
    parts.append(f"Experience: {p.experience_years:g} years")
    if p.projects:
        projs = (f"{x.name} ({', '.join(x.technologies[:5])})" for x in p.projects[:4])
        parts.append("Projects: " + "; ".join(projs))
    if p.education:
        edu = (f"{e.degree} {e.institution}".strip() for e in p.education[:2])
        parts.append("Education: " + "; ".join(edu))
    if p.preferred_locations:
        parts.append("Locations: " + ", ".join(p.preferred_locations))
    return ". ".join(parts)


def candidate_skills(p: ProfileSchema) -> set[str]:
    """Profile skills mapped onto the same canonical vocabulary the jobs index uses."""
    raw = " , ".join(p.skills + [t for pr in p.projects for t in pr.technologies])
    return {s.lower() for s in extract_skills(raw, limit=200)} | {s.lower() for s in p.skills}


def skill_gap(p: ProfileSchema, job: Job) -> tuple[list[str], list[str]]:
    have = candidate_skills(p)
    matched = [s for s in (job.skills or []) if s.lower() in have]
    missing = [s for s in (job.skills or []) if s.lower() not in have]
    return matched, missing


def heuristic_fit(matched: list[str], job: Job, hit_score: float | None = None) -> int:
    """Deterministic 0-100 fit used when no LLM score is available (and by the agent tool)."""
    total = max(len(job.skills or []), 1)
    return round(100 * len(matched) / total) if job.skills else 40


EXPLAIN_SYSTEM = (
    "You are a career coach. For each job, write ONE or TWO sentences explaining why it does or "
    "does not fit this candidate, using ONLY facts present in the candidate profile and the job "
    "posting provided. Mention concrete matching skills/experience. Be honest about gaps. "
    "fit_score is 0-100 (90+ = excellent, 60-89 = good, 30-59 = partial, <30 = poor)."
)


async def explain_matches(
    session: AsyncSession, llm, profile: ProfileSchema, hits: list[Hit]
) -> list[MatchOut]:
    llm = llm or get_llm()
    facts = []
    for h in hits:
        matched, missing = skill_gap(profile, h.job)
        facts.append((h, matched, missing))

    async def produce() -> dict:
        jobs_txt = "\n\n".join(
            f"[job_id={h.job.id}] {h.job.title} at {h.job.company} ({h.job.location})\n"
            f"Required skills: {', '.join(h.job.skills or []) or 'n/a'}\n"
            f"Candidate already has: {', '.join(m) or 'none of these'}\n"
            f"Description: {h.job.description[:700]}"
            for h, m, _ in facts
        )
        prompt = (
            f"Candidate profile:\n{profile_query_text(profile)}\n\nJobs:\n{jobs_txt}\n\n"
            "Return one item per job_id."
        )
        res = await llm.generate(prompt, system=EXPLAIN_SYSTEM, schema=ExplainOut)
        return ExplainOut.model_validate_json(res.text).model_dump()

    key = prompt_key("explain-v1", profile.model_dump(), [h.job.id for h in hits])
    try:
        data = await asyncio.wait_for(
            cached_json(session, key, produce), get_settings().explain_budget_s
        )
        by_id = {i["job_id"]: i for i in data["items"]}
    except Exception:  # noqa: BLE001 - explanations degrade gracefully, ranking still returned
        log.warning("explanation failed, using heuristic", exc_info=True)
        by_id = {}

    out = []
    for h, matched, missing in facts:
        e = by_id.get(h.job.id)
        out.append(
            MatchOut(
                job=job_out(h),
                why_it_matches=e["why_it_matches"] if e else
                f"Matches on {', '.join(matched[:4]) or 'overall profile similarity'}.",
                matched_skills=matched,
                missing_skills=missing,  # deterministic, so it can never be hallucinated
                fit_score=e["fit_score"] if e else heuristic_fit(matched, h.job),
            )
        )
    return out


async def build_query_vector(
    session: AsyncSession, embedder, profile_row: Profile, text_: str
) -> list[float]:
    q0 = (await embed_cached(session, embedder, [text_], QUERY))[0]
    delta = [float(x) for x in profile_row.pref_vector] if profile_row.pref_vector is not None else None
    return apply_preference(q0, delta)


async def get_matches(
    session: AsyncSession, embedder, llm, user_id: int, *, k: int = 10,
    filters: SearchFilters | None = None, personalise: bool = True, explain: bool = True,
) -> list[MatchOut]:
    row = await session.get(Profile, user_id)
    if row is None:
        return []
    profile = ProfileSchema.model_validate(row.data)
    text_ = profile_query_text(profile)
    if personalise:
        qvec = await build_query_vector(session, embedder, row, text_)
    else:
        qvec = (await embed_cached(session, embedder, [text_], QUERY))[0]
    disliked = (
        await session.execute(
            select(Feedback.job_id).where(Feedback.user_id == user_id, Feedback.liked.is_(False))
        )
    ).scalars().all()
    f = filters or SearchFilters()
    f.exclude_ids = list({*f.exclude_ids, *disliked})
    hits = await search(
        session, embedder, text_, filters=f, mode="hybrid_rerank", limit=k, qvec=qvec, llm=llm
    )
    if not explain:
        return [
            MatchOut(job=job_out(h), why_it_matches="", matched_skills=skill_gap(profile, h.job)[0],
                     missing_skills=skill_gap(profile, h.job)[1], fit_score=0)
            for h in hits
        ]
    return await explain_matches(session, llm, profile, hits)
