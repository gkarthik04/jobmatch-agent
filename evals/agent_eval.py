"""Agent eval: 15 scripted multi-turn conversations against the real agent, tools and database.

Per turn we check: expected tools were called in order, with the right arguments (including that
"the 2nd one" resolved to the correct job_id from the previous results), the task completed
(non-empty answer, no step-limit hit, no model outage), and the step count stayed within the limit.
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import get_sessionmaker
from app.models import Message, Profile, SavedJob, User
from app.schemas import Education, ProfileSchema, Project
from app.security import hash_password
from app.services.agent import run_agent
from app.services.agent_tools import AgentContext
from evals.metrics import mean
from evals.retrieval_eval import load_golden

PROFILE = ProfileSchema(
    name="Eval Candidate", skills=["Python", "SQL", "Docker", "FastAPI", "PostgreSQL", "Git"],
    experience_years=1.5, education=[Education(degree="B.Tech CSE", institution="Anna University", year=2025)],
    projects=[Project(name="RAG bot", description="retrieval QA", technologies=["FAISS", "Python"])],
    preferred_roles=["Backend Engineer", "Data Analyst"], preferred_locations=["Austin, TX"],
)


@dataclass
class Turn:
    say: str
    tools: list[str] = field(default_factory=list)  # expected tools, in order (extras allowed between)
    args: list[tuple[str, dict]] = field(default_factory=list)  # (tool, {arg: matcher})
    no_tools: bool = False
    expect_tool_failure: bool = False


# matchers: literal | ("pos", n) job id at position n of the results the agent was looking at
# (previous turn's, or this turn's own search when it searches first) | ("has", substr) |
# ("any", [substr, ...]) | ("fixed", id)
CONVERSATIONS: list[tuple[str, list[Turn]]] = [
    ("basic search", [Turn("Find python backend developer jobs", ["search_jobs"],
                           [("search_jobs", {"query": ("has", "python")})])]),
    ("remote filter", [Turn("Show me remote data analyst jobs", ["search_jobs"],
                            [("search_jobs", {"remote": True, "query": ("has", "analyst")})])]),
    ("follow-up: only remote", [
        Turn("Find data analyst jobs", ["search_jobs"], [("search_jobs", {"query": ("has", "analyst")})]),
        Turn("only remote ones", ["search_jobs"],
             [("search_jobs", {"remote": True, "query": ("has", "analyst")})])]),
    ("follow-up: compare 2nd with resume", [
        Turn("Find software engineer jobs", ["search_jobs"]),
        Turn("compare the 2nd one with my resume", ["compare_resume_to_job"],
             [("compare_resume_to_job", {"job_id": ("pos", 2)})])]),
    ("follow-up: save first", [
        Turn("Find devops engineer jobs", ["search_jobs"]),
        Turn("save the first one", ["save_job"], [("save_job", {"job_id": ("pos", 1)})])]),
    ("follow-up: details of 3rd", [
        Turn("Find customer service jobs", ["search_jobs"]),
        Turn("give me the details of the third job", ["get_job_details"],
             [("get_job_details", {"job_id": ("pos", 3)})])]),
    ("save then list saved", [
        Turn("Find accountant jobs", ["search_jobs"]),
        Turn("save the second one", ["save_job"], [("save_job", {"job_id": ("pos", 2)})]),
        Turn("what jobs have I saved?", ["list_saved_jobs"])]),
    ("location filter", [Turn("Find sales jobs in Chicago", ["search_jobs"],
                              [("search_jobs", {"location": ("has", "chicago")})])]),
    ("recency filter", [Turn("software engineer jobs posted in the last 7 days", ["search_jobs"],
                             [("search_jobs", {"posted_within_days": 7})])]),
    ("unknown job id recovers gracefully", [
        Turn("show me the details of job 99999999", ["get_job_details"],
             [("get_job_details", {"job_id": ("fixed", 99999999)})], expect_tool_failure=True)]),
    ("chit-chat needs no tools", [Turn("hi! what can you help me with?", no_tools=True)]),
    ("multi-step: search then compare", [
        Turn("Find data engineer jobs and tell me how my resume compares to the top one",
             ["search_jobs", "compare_resume_to_job"],
             [("compare_resume_to_job", {"job_id": ("pos", 1)})])]),
    ("save two jobs", [
        Turn("Find marketing jobs", ["search_jobs"]),
        Turn("save the first two", ["save_job", "save_job"],
             [("save_job", {"job_id": ("pos", 1)}), ("save_job", {"job_id": ("pos", 2)})])]),
    ("refine location", [
        Turn("Find nurse or healthcare jobs", ["search_jobs"]),
        Turn("now only in Texas", ["search_jobs"], [("search_jobs", {"location": ("any", ["tx", "texas"])})])]),
    ("compare by explicit id", [Turn("compare job {JOB_ID} with my resume", ["compare_resume_to_job"],
                                     [("compare_resume_to_job", {"job_id": ("fixed", "{JOB_ID}")})])]),
]


def _match(actual, expected, prev_results: list[int], job_id: int) -> bool:
    if isinstance(expected, tuple):
        kind, val = expected
        if kind == "pos":
            return len(prev_results) >= val and actual == prev_results[val - 1]
        if kind == "has":
            return actual is not None and str(val).lower() in str(actual).lower()
        if kind == "any":
            return actual is not None and any(v in str(actual).lower() for v in val)
        if kind == "fixed":
            return actual == (job_id if val == "{JOB_ID}" else val)
    return actual == expected


def _subsequence(expected: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(any(e == a for a in it) for e in expected)


async def _prev_results(user_id: int) -> list[int]:
    async with get_sessionmaker()() as s:
        meta = (await s.execute(
            select(Message.meta).where(Message.user_id == user_id, Message.role == "assistant")
            .order_by(Message.id.desc()).limit(1))).scalar()
    return (meta or {}).get("state", {}).get("last_results", [])


async def run(embedder, llm, n: int | None = None) -> dict:
    settings = get_settings()
    job_id = 0
    async with get_sessionmaker()() as s:  # a real, valid job id for the explicit-id scenario
        from app.models import Job

        g = load_golden()["cases"][0]["relevant_source_ids"][0]
        job_id = (await s.execute(select(Job.id).where(Job.source_id == g))).scalar()
    results = []
    for name, turns in CONVERSATIONS[:n]:
        async with get_sessionmaker()() as s:
            user = User(email=f"eval-agent-{uuid.uuid4().hex[:8]}@example.com", password_hash=hash_password("x" * 12))
            s.add(user)
            await s.commit()
            s.add(Profile(user_id=user.id, data=PROFILE.model_dump(), resume_text=""))
            await s.commit()
        uid = user.id
        conv = {"name": name, "turns": []}
        for t in turns:
            prev = await _prev_results(uid)
            say = t.say.replace("{JOB_ID}", str(job_id))
            async with get_sessionmaker()() as s:
                ctx = AgentContext(session=s, user_id=uid, embedder=embedder, llm=llm)
                r = await run_agent(ctx, say)
            calls = r.tool_calls
            names = [c.name for c in calls]
            if "search_jobs" in names:  # same-turn dependency: "the top one" = this turn's own results
                prev = await _prev_results(uid)
            if t.no_tools:
                tools_ok = not names
            else:
                tools_ok = _subsequence(t.tools, names)
            args_ok = all(
                any(c.name == tool and all(_match((c.args or {}).get(k), v, prev, job_id) for k, v in cons.items())
                    for c in calls)
                for tool, cons in t.args
            )
            llm_calls = sum(1 for st in r.steps if st.kind == "llm")
            completed = bool(r.reply.strip()) and not r.hit_step_limit and all(st.ok for st in r.steps if st.kind == "llm")
            if t.expect_tool_failure:
                completed = completed and any(not c.ok for c in calls)
            conv["turns"].append({
                "say": say, "tools_called": names, "tools_ok": tools_ok, "args_ok": args_ok,
                "completed": completed, "within_limit": llm_calls <= settings.agent_max_steps,
                "llm_calls": llm_calls, "latency_ms": sum(st.latency_ms for st in r.steps),
                "tokens": sum(st.prompt_tokens + st.output_tokens for st in r.steps),
                "reply": r.reply[:160],
            })
        conv["passed"] = all(tt["tools_ok"] and tt["args_ok"] and tt["completed"] and tt["within_limit"]
                             for tt in conv["turns"])
        results.append(conv)
        async with get_sessionmaker()() as s:  # cleanup eval user (cascades messages/saved/steps)
            await s.execute(delete(SavedJob).where(SavedJob.user_id == uid))
            await s.execute(delete(User).where(User.id == uid))
            await s.commit()
    turns = [t for c in results for t in c["turns"]]
    return {
        "n_conversations": len(results), "n_turns": len(turns),
        "conversations_passed": mean([float(c["passed"]) for c in results]),
        "tool_selection": mean([float(t["tools_ok"]) for t in turns]),
        "argument_accuracy": mean([float(t["args_ok"]) for t in turns]),
        "task_completion": mean([float(t["completed"]) for t in turns]),
        "within_step_limit": mean([float(t["within_limit"]) for t in turns]),
        "mean_llm_calls_per_turn": mean([t["llm_calls"] for t in turns]),
        "mean_latency_ms": mean([t["latency_ms"] for t in turns]),
        "mean_tokens_per_turn": mean([t["tokens"] for t in turns]),
        "conversations": results,
    }
