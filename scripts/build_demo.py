"""Generate the real data shown in the landing page demos (no invented results).

    python scripts/build_demo.py            # writes app/static/demo/*.json

* personas.json   - top hybrid-search results (open jobs only) from the real index for four sample resumes,
                    with deterministic matched / missing skills and a skill overlap percentage.
* agent.json      - real traces of the agent loop (tools, arguments, latency, reply excerpt).
* ticker.json     - real job titles and skills for the marquee.
"""

import asyncio
import json
import random
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, ".")

from sqlalchemy import delete, text  # noqa: E402

from app.db import dispose_engine, get_sessionmaker  # noqa: E402
from app.models import Profile, User  # noqa: E402
from app.schemas import Education, ProfileSchema, Project  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.agent import run_agent  # noqa: E402
from app.services.agent_tools import AgentContext  # noqa: E402
from app.services.embeddings import make_embedder  # noqa: E402
from app.services.llm import get_llm  # noqa: E402
from app.services.matching import profile_query_text, skill_gap  # noqa: E402
from app.services.search import SearchFilters, dedupe_hits, search  # noqa: E402
from app.services.skills import all_skill_names  # noqa: E402

OUT = Path("app/static/demo")

PERSONAS = [
    ("data", "Data analyst", ProfileSchema(
        name="Sample", skills=["SQL", "Power BI", "Excel", "Tableau", "Statistics"], experience_years=1.5,
        education=[Education(degree="B.Tech CSE", institution="", year=2025)],
        projects=[Project(name="Sales dashboard", description="", technologies=["Tableau", "SQL"])],
        preferred_roles=["Data Analyst"], preferred_locations=[])),
    ("backend", "Backend engineer", ProfileSchema(
        name="Sample", skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"], experience_years=2,
        education=[Education(degree="B.E. IT", institution="", year=2024)], projects=[],
        preferred_roles=["Backend Engineer"], preferred_locations=[])),
    ("frontend", "Frontend developer", ProfileSchema(
        name="Sample", skills=["JavaScript", "React", "TypeScript", "HTML", "CSS"], experience_years=1,
        education=[Education(degree="B.Tech CSE", institution="", year=2025)], projects=[],
        preferred_roles=["Frontend Developer"], preferred_locations=[])),
    ("ml", "Data scientist", ProfileSchema(
        name="Sample", skills=["Python", "SQL", "Machine learning", "PyTorch", "Statistics"], experience_years=2,
        education=[Education(degree="M.Tech AI", institution="", year=2024)], projects=[],
        preferred_roles=["Data Scientist"], preferred_locations=[])),
]

AGENT_SCENARIOS = [
    ("search", "Search and refine", ["Find remote data analyst jobs", "compare the 2nd one with my resume"]),
    ("save", "Save and recall", ["Find python backend developer jobs", "save the first one", "what jobs have I saved?"]),
    ("filter", "Filters from follow ups", ["Show me product manager jobs", "only remote ones"]),
]


def tidy(s: str) -> str:
    s = re.sub(r"\*\*|__", "", s)
    s = s.replace(" - ", ", ").replace("–", ",").replace("—", ",")
    return re.sub(r"[ \t]+", " ", s).strip()


def excerpt(reply: str, limit: int = 330) -> str:
    reply = tidy(reply)
    if len(reply) <= limit:
        return reply
    cut = reply[:limit]
    return cut[: max(cut.rfind("\n"), cut.rfind(". "), 120)].rstrip() + " ..."


async def build_personas(embedder) -> list[dict]:
    out = []
    async with get_sessionmaker()() as s:
        for pid, label, prof in PERSONAS:
            q = profile_query_text(prof)
            hits = await search(s, embedder, q, filters=SearchFilters(open_only=True), mode="hybrid", limit=24)
            hits = dedupe_hits(hits, limit=4)
            rows = []
            for h in hits:
                matched, missing = skill_gap(prof, h.job)
                total = max(len(h.job.skills or []), 1)
                rows.append({
                    "title": tidy(h.job.title), "company": h.job.company or "Company not listed",
                    "location": h.job.location or "", "remote": bool(h.job.remote),
                    "matched": matched[:5], "missing": missing[:3],
                    "overlap": round(100 * len(matched) / total),
                })
            out.append({"id": pid, "label": label, "years": prof.experience_years, "skills": prof.skills,
                        "role": prof.preferred_roles[0], "results": rows})
    return out


async def build_agent(embedder, llm) -> list[dict]:
    traces = []
    prof = PERSONAS[0][2].model_copy(update={"name": "Demo"})
    for sid, label, turns in AGENT_SCENARIOS:
        async with get_sessionmaker()() as s:
            user = User(email=f"demo-{uuid.uuid4().hex[:8]}@example.com", password_hash=hash_password("x" * 12))
            s.add(user)
            await s.commit()
            s.add(Profile(user_id=user.id, data=prof.model_dump(), resume_text=""))
            await s.commit()
        uid, conv = user.id, []
        for say in turns:
            async with get_sessionmaker()() as s:
                ctx = AgentContext(session=s, user_id=uid, embedder=embedder, llm=llm)
                r = await run_agent(ctx, say)
            tools = [{"name": t.name, "args": t.args, "ok": t.ok, "ms": round(t.latency_ms)} for t in r.tool_calls]
            conv.append({"user": say, "tools": tools, "reply": excerpt(r.reply),
                         "llm_calls": sum(1 for st in r.steps if st.kind == "llm"),
                         "total_ms": round(sum(st.latency_ms for st in r.steps))})
            print(sid, say, [t["name"] for t in tools], flush=True)
        traces.append({"id": sid, "label": label, "turns": conv})
        async with get_sessionmaker()() as s:
            await s.execute(delete(User).where(User.id == uid))
            await s.commit()
    return traces


async def build_ticker() -> dict:
    async with get_sessionmaker()() as s:
        rows = (await s.execute(text(
            "SELECT title FROM jobs WHERE source <> 'linkedin_2024' AND last_seen_at > now() - interval '3 days' AND length(title) < 40 AND title ~ '^[A-Za-z0-9 ,&/()+.]+$' "
            "ORDER BY md5(source_id) LIMIT 400"))).scalars().all()
    seen, titles = set(), []
    for t in rows:
        k = t.strip().lower()
        if k not in seen and not re.search(r"\$|\d{3}|\bPR\d", t):
            seen.add(k)
            titles.append(tidy(t))
    random.Random(7).shuffle(titles)
    skills = [s.title() if len(s) > 3 else s.upper() for s in all_skill_names()]
    return {"titles": titles[:24], "skills": skills[:28]}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    embedder, llm = make_embedder(), get_llm()
    (OUT / "personas.json").write_text(json.dumps(await build_personas(embedder), indent=1), encoding="utf-8")
    (OUT / "ticker.json").write_text(json.dumps(await build_ticker(), indent=1), encoding="utf-8")
    (OUT / "agent.json").write_text(json.dumps(await build_agent(embedder, llm), indent=1), encoding="utf-8")
    await dispose_engine()
    print("demo data written to", OUT)


if __name__ == "__main__":
    asyncio.run(main())
