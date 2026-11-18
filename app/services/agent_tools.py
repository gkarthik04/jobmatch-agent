"""Agent tools: Pydantic-validated arguments, async handlers, Gemini function declarations."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job, SavedJob
from app.schemas import ProfileSchema
from app.services.matching import heuristic_fit, skill_gap
from app.services.search import SearchFilters, dedupe_hits, search


@dataclass
class AgentContext:
    session: AsyncSession
    user_id: int
    embedder: Any
    llm: Any
    profile: ProfileSchema | None = None
    # conversational state carried across turns (persisted in messages.meta)
    state: dict = field(default_factory=lambda: {"last_query": None, "last_filters": {}, "last_results": []})


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300,
                       description="Free-text job search, e.g. 'python backend engineer'")
    remote: bool | None = Field(None, description="true = remote jobs only")
    location: str | None = Field(None, description="City/state/country substring, e.g. 'Austin'")
    posted_within_days: int | None = Field(None, ge=1, le=365, description="Only recent postings")
    limit: int = Field(5, ge=1, le=10, description="How many results (default 5)")


class JobIdArgs(BaseModel):
    job_id: int = Field(ge=1, description="The job_id from a previous search result")


class NoArgs(BaseModel):
    pass


class ToolError(Exception):
    """Expected failure the model can recover from (bad id, no profile, ...)."""


def _brief(j: Job) -> dict:
    return {
        "job_id": j.id, "title": j.title, "company": j.company, "location": j.location,
        "remote": j.remote, "skills": (j.skills or [])[:8],
    }


async def search_jobs(ctx: AgentContext, a: SearchArgs) -> dict:
    filters = SearchFilters(location=a.location, remote=a.remote, posted_within_days=a.posted_within_days)
    hits = await search(
        ctx.session, ctx.embedder, a.query, filters=filters, mode="hybrid", limit=a.limit * 3
    )
    hits = dedupe_hits(hits, limit=a.limit)
    ctx.state["last_query"] = a.query
    ctx.state["last_filters"] = {
        k: v for k, v in {"remote": a.remote, "location": a.location,
                          "posted_within_days": a.posted_within_days}.items() if v is not None
    }
    ctx.state["last_results"] = [h.job.id for h in hits]
    return {"count": len(hits), "results": [{"position": i + 1, **_brief(h.job)} for i, h in enumerate(hits)]}


async def get_job_details(ctx: AgentContext, a: JobIdArgs) -> dict:
    job = await ctx.session.get(Job, a.job_id)
    if job is None:
        raise ToolError(f"No job with job_id={a.job_id}")
    return {**_brief(job), "experience_level": job.experience_level,
            "posted_at": job.posted_at.isoformat() if job.posted_at else None,
            "description": job.description[:1800]}


async def compare_resume_to_job(ctx: AgentContext, a: JobIdArgs) -> dict:
    if ctx.profile is None:
        raise ToolError("The user has not uploaded a resume yet, so there is no profile to compare")
    job = await ctx.session.get(Job, a.job_id)
    if job is None:
        raise ToolError(f"No job with job_id={a.job_id}")
    matched, missing = skill_gap(ctx.profile, job)
    return {"job_id": job.id, "title": job.title, "matched_skills": matched,
            "missing_skills": missing, "fit_score": heuristic_fit(matched, job)}


async def save_job(ctx: AgentContext, a: JobIdArgs) -> dict:
    job = await ctx.session.get(Job, a.job_id)
    if job is None:
        raise ToolError(f"No job with job_id={a.job_id}")
    await ctx.session.execute(
        pg_insert(SavedJob).values(user_id=ctx.user_id, job_id=job.id).on_conflict_do_nothing()
    )
    await ctx.session.commit()
    return {"saved": True, "job_id": job.id, "title": job.title}


async def list_saved_jobs(ctx: AgentContext, a: NoArgs) -> dict:
    rows = (
        await ctx.session.execute(
            select(Job).join(SavedJob, SavedJob.job_id == Job.id)
            .where(SavedJob.user_id == ctx.user_id).order_by(SavedJob.id.desc()).limit(50)
        )
    ).scalars().all()
    return {"count": len(rows), "saved": [_brief(j) for j in rows]}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[AgentContext, Any], Awaitable[dict]]


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("search_jobs", "Search the live jobs index (hybrid keyword + semantic). Use for any request "
             "to find, list or refine jobs.", SearchArgs, search_jobs),
        Tool("get_job_details", "Get the full description and metadata of one job by job_id.",
             JobIdArgs, get_job_details),
        Tool("compare_resume_to_job", "Compare the user's resume profile to a job: matched skills, "
             "missing skills and a 0-100 fit score.", JobIdArgs, compare_resume_to_job),
        Tool("save_job", "Save a job to the user's saved list.", JobIdArgs, save_job),
        Tool("list_saved_jobs", "List the jobs the user has saved.", NoArgs, list_saved_jobs),
    ]
}


def _prop_schema(prop: dict):
    from google.genai import types

    if "anyOf" in prop:  # Optional[X] -> X
        non_null = [p for p in prop["anyOf"] if p.get("type") != "null"]
        prop = {**non_null[0], **{k: v for k, v in prop.items() if k != "anyOf"}}
    kind = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}[prop["type"]]
    return types.Schema(type=kind, description=prop.get("description"))


def gemini_tools() -> list:
    from google.genai import types

    decls = []
    for t in TOOLS.values():
        js = t.args_model.model_json_schema()
        props = {k: _prop_schema(v) for k, v in js.get("properties", {}).items()}
        params = (
            types.Schema(type="OBJECT", properties=props, required=js.get("required", []))
            if props else None
        )
        decls.append(types.FunctionDeclaration(name=t.name, description=t.description, parameters=params))
    return [types.Tool(function_declarations=decls)]
