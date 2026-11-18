"""LLM-as-judge for match explanations: groundedness and relevance (1-5), hand-written judge.

For a handful of synthetic candidate personas we generate real matches (`get_matches`) and ask a
judge call to grade each explanation against ONLY the job posting + candidate profile text.
Caveat (stated in the report): the judge is the same model family as the generator.
"""

import uuid

from pydantic import BaseModel, Field
from sqlalchemy import delete

from app.db import get_sessionmaker
from app.models import Profile, User
from app.schemas import Education, ProfileSchema, Project
from app.security import hash_password
from app.services.matching import get_matches, profile_query_text
from evals.metrics import mean

PERSONAS = [
    ProfileSchema(name="A", skills=["Python", "SQL", "Tableau", "Excel", "Statistics"], experience_years=1.0,
                  education=[Education(degree="B.Tech CSE", institution="VIT", year=2025)],
                  projects=[Project(name="Sales dashboard", description="BI dashboard", technologies=["Tableau"])],
                  preferred_roles=["Data Analyst"], preferred_locations=[]),
    ProfileSchema(name="B", skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Git"], experience_years=2.5,
                  education=[Education(degree="B.E. IT", institution="NIT", year=2023)],
                  projects=[Project(name="URL shortener", description="API", technologies=["FastAPI", "Redis"])],
                  preferred_roles=["Backend Engineer"], preferred_locations=["Austin, TX"]),
    ProfileSchema(name="C", skills=["Accounting", "Excel", "QuickBooks", "Financial Analysis"], experience_years=4,
                  education=[Education(degree="B.Com", institution="Delhi University", year=2020)], projects=[],
                  preferred_roles=["Accountant"], preferred_locations=["Chicago, IL"]),
    ProfileSchema(name="D", skills=["Customer Service", "Communication", "Salesforce", "Sales"], experience_years=3,
                  education=[Education(degree="BBA", institution="Pune University", year=2021)], projects=[],
                  preferred_roles=["Customer Success Manager"], preferred_locations=[]),
    ProfileSchema(name="E", skills=["Docker", "Kubernetes", "Terraform", "AWS", "Linux", "Python"], experience_years=3.5,
                  education=[Education(degree="B.Tech ECE", institution="BITS", year=2022)], projects=[],
                  preferred_roles=["DevOps Engineer"], preferred_locations=["Remote"]),
    ProfileSchema(name="F", skills=["Marketing", "SEO", "Content", "Excel"], experience_years=2,
                  education=[Education(degree="BBA Marketing", institution="Anna University", year=2022)], projects=[],
                  preferred_roles=["Digital Marketing Specialist"], preferred_locations=[]),
]


class JudgeItem(BaseModel):
    job_id: int
    groundedness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    unsupported_claims: list[str] = []


class JudgeOut(BaseModel):
    items: list[JudgeItem]


JUDGE_SYSTEM = (
    "You are a strict evaluator of job-match explanations. Judge ONLY against the candidate profile "
    "and job posting text provided; do not use outside knowledge.\n"
    "groundedness (1-5): 5 = every factual claim is supported by the profile/posting; 3 = some claims "
    "are unsupported or vague; 1 = mostly invented or contradicts the text.\n"
    "relevance (1-5): 5 = the explanation clearly says why this job fits (or does not fit) this "
    "candidate with concrete skills/experience; 3 = generic; 1 = off-topic.\n"
    "List any unsupported claims verbatim in unsupported_claims."
)


async def run(embedder, llm, k: int = 5, n_personas: int | None = None) -> dict:
    rows = []
    for p in PERSONAS[:n_personas]:
        async with get_sessionmaker()() as s:
            user = User(email=f"eval-judge-{uuid.uuid4().hex[:8]}@example.com", password_hash=hash_password("x" * 12))
            s.add(user)
            await s.commit()
            s.add(Profile(user_id=user.id, data=p.model_dump(), resume_text=""))
            await s.commit()
            matches = await get_matches(s, embedder, llm, user.id, k=k)
        async with get_sessionmaker()() as s:
            await s.execute(delete(User).where(User.id == user.id))
            await s.commit()
        blocks = "\n\n".join(
            f"[job_id={m.job.id}] {m.job.title} at {m.job.company} ({m.job.location})\n"
            f"Skills listed: {', '.join(m.job.skills)}\nPosting text: {m.job.snippet}\n"
            f"EXPLANATION TO JUDGE: {m.why_it_matches}"
            for m in matches
        )
        prompt = f"Candidate profile:\n{profile_query_text(p)}\n\nJobs and explanations:\n{blocks}"
        res = await llm.generate(prompt, system=JUDGE_SYSTEM, schema=JudgeOut)
        judged = JudgeOut.model_validate_json(res.text).items
        by_id = {j.job_id: j for j in judged}
        for m in matches:
            j = by_id.get(m.job.id)
            if j:
                rows.append({"persona": p.name, "title": m.job.title, "explanation": m.why_it_matches,
                             "groundedness": j.groundedness, "relevance": j.relevance,
                             "unsupported": j.unsupported_claims, "fit_score": m.fit_score})
    return {
        "n_explanations": len(rows),
        "groundedness_mean": mean([r["groundedness"] for r in rows]),
        "relevance_mean": mean([r["relevance"] for r in rows]),
        "grounded_ge4": mean([float(r["groundedness"] >= 4) for r in rows]),
        "relevant_ge4": mean([float(r["relevance"] >= 4) for r in rows]),
        "with_unsupported_claims": mean([float(bool(r["unsupported"])) for r in rows]),
        "rows": rows,
    }
