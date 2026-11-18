from datetime import datetime

from pydantic import BaseModel

from app.services.job_status import is_open, link_for, source_label


class JobOut(BaseModel):
    id: int
    title: str
    company: str | None
    location: str | None
    remote: bool
    skills: list[str]
    posted_at: datetime | None
    snippet: str
    score: float
    ranks: dict[str, int] = {}
    rerank_score: float | None = None
    url: str | None = None  # the listing page
    apply_url: str | None = None  # where to apply (falls back to url)
    source: str = ""
    source_label: str = ""
    is_open: bool = False
    expires_at: datetime | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[JobOut]


def job_out(hit) -> JobOut:
    j = hit.job
    return JobOut(
        id=j.id, title=j.title, company=j.company, location=j.location, remote=j.remote,
        skills=list(j.skills or []), posted_at=j.posted_at,
        snippet=(j.description or "")[:300].replace("\n", " "),
        score=round(hit.score, 5), ranks=hit.ranks,
        rerank_score=hit.rerank_score,
        url=j.url, apply_url=link_for(j), source=j.source or "",
        source_label=source_label(j.source), is_open=is_open(j), expires_at=j.expires_at,
    )


# ---- resume / profile -------------------------------------------------------------------------
class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: int | None = None


class Project(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = []


class ProfileSchema(BaseModel):
    """What Gemini must return for a resume. Validated; invalid output is retried with feedback."""

    name: str
    skills: list[str]
    experience_years: float
    education: list[Education]
    projects: list[Project]
    preferred_roles: list[str]
    preferred_locations: list[str]
