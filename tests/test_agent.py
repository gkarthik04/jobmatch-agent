import asyncio

import pytest
from pydantic import BaseModel

from app.services import agent as agent_mod
from app.services.agent import check_observation, execute_tool, run_agent
from app.services.agent_tools import TOOLS, AgentContext, JobIdArgs, Tool, ToolError, gemini_tools
from app.services.llm import FunctionCall, LLMResult


class FakeLLM:
    """Scripted model: each generate() pops the next LLMResult. Records whether tools were offered."""

    def __init__(self, script):
        self.script, self.tools_offered = list(script), []

    async def generate(self, contents, *, system=None, schema=None, tools=None, temperature=0.0):
        self.tools_offered.append(bool(tools))
        return self.script.pop(0)


def call(name, **args):
    return LLMResult(function_calls=[FunctionCall(name, args)], prompt_tokens=10, output_tokens=5)


def say(text):
    return LLMResult(text=text, prompt_tokens=10, output_tokens=5)


@pytest.fixture
def ctx(monkeypatch):
    async def fake_memory(c):
        return [], []

    async def fake_persist(c, msg, res):
        return None

    monkeypatch.setattr(agent_mod, "load_memory", fake_memory)
    monkeypatch.setattr(agent_mod, "_persist", fake_persist)
    return AgentContext(session=None, user_id=1, embedder=None, llm=None)


@pytest.fixture(autouse=True)
def echo_tools(monkeypatch):
    async def echo(c, a):
        return {"job_id": a.job_id, "ok": True}

    async def boom(c, a):
        raise RuntimeError("db exploded")

    async def slow(c, a):
        await asyncio.sleep(5)
        return {}

    async def missing(c, a):
        raise ToolError("No job with job_id=999")

    tools = dict(TOOLS)
    tools.update({
        "echo": Tool("echo", "t", JobIdArgs, echo),
        "boom": Tool("boom", "t", JobIdArgs, boom),
        "slow": Tool("slow", "t", JobIdArgs, slow),
        "missing": Tool("missing", "t", JobIdArgs, missing),
    })
    monkeypatch.setattr(agent_mod, "TOOLS", tools)


# ---- ACT: validation / timeout / failure recovery ---------------------------------------------
async def test_tool_runs_with_valid_args(ctx):
    res, log = await execute_tool(ctx, FunctionCall("echo", {"job_id": 7.0}), 2)
    assert res == {"job_id": 7, "ok": True} and log.ok


async def test_invalid_args_become_error_observation_not_exception(ctx):
    res, log = await execute_tool(ctx, FunctionCall("echo", {"job_id": "abc"}), 2)
    assert "Invalid arguments" in res["error"] and not log.ok
    res, _ = await execute_tool(ctx, FunctionCall("echo", {}), 2)
    assert "job_id" in res["error"]


async def test_unknown_tool_is_reported(ctx):
    res, log = await execute_tool(ctx, FunctionCall("hack_the_planet", {}), 2)
    assert "Unknown tool" in res["error"] and not log.ok


async def test_timeout_is_enforced(ctx):
    res, log = await execute_tool(ctx, FunctionCall("slow", {"job_id": 1}), 0.05)
    assert "timed out" in res["error"] and log.error == "timeout"


async def test_crashing_tool_is_contained(ctx):
    class S:
        async def rollback(self):
            pass

    ctx.session = S()
    res, log = await execute_tool(ctx, FunctionCall("boom", {"job_id": 1}), 2)
    assert res == {"error": "Internal tool error"} and log.error == "RuntimeError"


# ---- CHECK -----------------------------------------------------------------------------------
def test_check_flags_errors_empty_results_and_duplicates():
    seen = set()
    assert "failed" in check_observation("echo", {"a": 1}, {"error": "x"}, seen)
    assert "No results" in check_observation("search_jobs", {"q": 1}, {"count": 0}, seen)
    assert check_observation("echo", {"a": 2}, {"ok": 1}, seen) is None
    assert "exact call" in check_observation("echo", {"a": 2}, {"ok": 1}, seen)


# ---- the loop --------------------------------------------------------------------------------
async def test_plain_answer_without_tools(ctx):
    ctx.llm = FakeLLM([say("Hello!")])
    r = await run_agent(ctx, "hi")
    assert r.reply == "Hello!" and r.tool_calls == [] and len(r.steps) == 1


async def test_tool_then_answer_logs_every_step(ctx):
    ctx.llm = FakeLLM([call("echo", job_id=3), say("Job 3 is fine.")])
    r = await run_agent(ctx, "check job 3")
    kinds = [(s.kind, s.name) for s in r.steps]
    assert kinds == [("llm", None), ("tool", "echo"), ("llm", None)]
    d = r.to_dict()
    assert d["usage"]["llm_calls"] == 2 and d["usage"]["prompt_tokens"] == 20


async def test_recovers_from_failed_tool_and_answers(ctx):
    ctx.llm = FakeLLM([call("missing", job_id=999), call("echo", job_id=1), say("Used job 1 instead.")])
    r = await run_agent(ctx, "x")
    assert [s.ok for s in r.tool_calls] == [False, True]
    assert r.reply == "Used job 1 instead."


async def test_step_limit_forces_final_answer_without_tools(ctx):
    ctx.llm = FakeLLM([call("echo", job_id=i) for i in range(1, 6)] + [say("Here is what I found.")])
    r = await run_agent(ctx, "loop forever", max_steps=6)
    assert sum(s.kind == "llm" for s in r.steps) == 6  # never more than max_steps model calls
    assert ctx.llm.tools_offered == [True] * 5 + [False]  # last call has tools disabled
    assert r.reply == "Here is what I found."


async def test_step_limit_when_model_still_wants_tools_at_the_end(ctx):
    ctx.llm = FakeLLM([call("echo", job_id=1), call("echo", job_id=2), call("echo", job_id=3)])
    r = await run_agent(ctx, "x", max_steps=3)
    assert r.hit_step_limit and r.reply  # always returns something user-facing


async def test_llm_outage_returns_graceful_message(ctx):
    class Down:
        async def generate(self, *a, **k):
            raise ConnectionError("down")

    ctx.llm = Down()
    r = await run_agent(ctx, "hi")
    assert "trouble reaching" in r.reply and not r.steps[0].ok


def test_gemini_declarations_cover_all_tools():
    names = {d.name for d in gemini_tools()[0].function_declarations}
    assert {"search_jobs", "get_job_details", "compare_resume_to_job", "save_job", "list_saved_jobs"} <= names
    search = next(d for d in gemini_tools()[0].function_declarations if d.name == "search_jobs")
    assert search.parameters.required == ["query"]
    assert issubclass(JobIdArgs, BaseModel)
