"""The agent loop, written out explicitly (no framework).

    PLAN   the model sees context + history and either answers or emits tool calls. Its first turn
           is asked to state a one-line plan, which we record.
    ACT    each tool call is validated against its Pydantic args model, run under a timeout, and
           any failure becomes a structured error observation instead of an exception.
    CHECK  `check_observation` (deterministic) inspects every result and appends a hint: tool
           failed / zero results / duplicate call. Then the observation goes back to the model,
           which decides to continue or answer.
    LIMIT  at most `max_steps` model calls. The last allowed call has tools disabled, so the
           agent always ends with a user-facing answer, never a dangling tool request.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError
from sqlalchemy import select

from app.config import get_settings
from app.models import AgentStep, Message, Profile, SavedJob
from app.schemas import ProfileSchema
from app.services.agent_tools import TOOLS, AgentContext, ToolError, gemini_tools
from app.services.llm import FunctionCall, LLMResult

log = logging.getLogger("app.agent")

HISTORY_MESSAGES = 12

SYSTEM = """You are JobMatch, a career agent that helps one user find and evaluate jobs from a live \
job index. You can call tools. Rules:
- Ground every claim about jobs in tool results; never invent jobs, companies or ids.
- When you need tools, first write ONE short sentence stating your plan, then call them.
- After tool results, check whether they answer the question; if not (e.g. zero results, error), \
adjust the arguments or approach once, otherwise tell the user plainly what went wrong.
- Resolve references like "the 2nd one" or "that job" using the 'Recent search results' list below \
(position -> job_id). For refinements like "only remote ones" or "in Austin", call search_jobs \
again with the previous query plus the new filter.
- Comparing a job to the user's resume = compare_resume_to_job. Saving = save_job.
- Be concise. List jobs as: position. Title - Company (Location) [job_id]."""


@dataclass
class StepLog:
    step: int
    kind: str  # llm | tool
    name: str | None = None
    args: dict | None = None
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    text: str | None = None  # model text on this step (plan for the first step)


@dataclass
class AgentResult:
    reply: str
    run_id: str
    steps: list[StepLog] = field(default_factory=list)
    hit_step_limit: bool = False

    @property
    def tool_calls(self) -> list[StepLog]:
        return [s for s in self.steps if s.kind == "tool"]

    def to_dict(self) -> dict:
        return {
            "reply": self.reply, "run_id": self.run_id, "hit_step_limit": self.hit_step_limit,
            "steps": [asdict(s) for s in self.steps],
            "usage": {
                "prompt_tokens": sum(s.prompt_tokens for s in self.steps),
                "output_tokens": sum(s.output_tokens for s in self.steps),
                "llm_calls": sum(1 for s in self.steps if s.kind == "llm"),
                "latency_ms": round(sum(s.latency_ms for s in self.steps)),
            },
        }


def check_observation(name: str, args: dict, result: dict, seen: set[str]) -> str | None:
    """CHECK phase: deterministic guidance attached to a tool observation."""
    key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    duplicate = key in seen
    seen.add(key)
    if "error" in result:
        return "The tool failed. Fix the arguments if that is the cause, otherwise explain to the user."
    if duplicate:
        return "You already made this exact call. Do not repeat it; answer using what you have."
    if name == "search_jobs" and result.get("count") == 0:
        return "No results. Relax a filter or rephrase the query once, or tell the user nothing matched."
    return None


async def execute_tool(ctx: AgentContext, call: FunctionCall, timeout_s: float) -> tuple[dict, StepLog]:
    """ACT phase for one call. Never raises: failures become {"error": ...} observations."""
    log_ = StepLog(step=0, kind="tool", name=call.name, args=call.args)
    start = time.perf_counter()
    tool = TOOLS.get(call.name)
    try:
        if tool is None:
            raise ToolError(f"Unknown tool '{call.name}'. Available: {', '.join(TOOLS)}")
        try:
            args = tool.args_model.model_validate(call.args)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ToolError(f"Invalid arguments: {msg}") from exc
        result = await asyncio.wait_for(tool.handler(ctx, args), timeout_s)
    except ToolError as exc:
        result, log_.ok, log_.error = {"error": str(exc)}, False, str(exc)
    except TimeoutError:
        result, log_.ok, log_.error = {"error": f"Tool timed out after {timeout_s:g}s"}, False, "timeout"
    except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the conversation
        log.exception("tool crashed", extra={"tool": call.name})
        try:
            await ctx.session.rollback()
        except Exception:  # noqa: BLE001
            pass
        result, log_.ok, log_.error = {"error": "Internal tool error"}, False, type(exc).__name__
    log_.latency_ms = (time.perf_counter() - start) * 1000
    return result, log_


def _context_block(ctx: AgentContext, saved_ids: list[int]) -> str:
    lines = []
    if ctx.profile:
        p = ctx.profile
        lines.append(
            f"User profile: {p.name}; {p.experience_years:g} yrs; skills: {', '.join(p.skills[:15])}; "
            f"target roles: {', '.join(p.preferred_roles)}; "
            f"locations: {', '.join(p.preferred_locations) or 'any'}"
        )
    else:
        lines.append("User has NOT uploaded a resume yet (resume comparison is unavailable).")
    lines.append(f"Saved jobs (job_ids): {saved_ids or 'none'}")
    st = ctx.state
    if st.get("last_results"):
        pos = ", ".join(f"{i + 1}->{jid}" for i, jid in enumerate(st["last_results"]))
        lines.append(f"Recent search results (position->job_id): {pos}")
        lines.append(f"Recent search query: {st.get('last_query')!r}, filters: {st.get('last_filters')}")
    return "\n".join(lines)


async def load_memory(ctx: AgentContext) -> tuple[list[tuple[str, str]], list[int]]:
    """Conversation history + persisted state + profile + saved jobs, all from Postgres."""
    rows = (
        await ctx.session.execute(
            select(Message).where(Message.user_id == ctx.user_id).order_by(Message.id.desc())
            .limit(HISTORY_MESSAGES)
        )
    ).scalars().all()[::-1]
    for m in reversed(rows):
        if m.role == "assistant" and m.meta and m.meta.get("state"):
            ctx.state.update(m.meta["state"])
            break
    prof = await ctx.session.get(Profile, ctx.user_id)
    ctx.profile = ProfileSchema.model_validate(prof.data) if prof else None
    saved = (
        await ctx.session.execute(select(SavedJob.job_id).where(SavedJob.user_id == ctx.user_id))
    ).scalars().all()
    return [(m.role, m.content) for m in rows], list(saved)


def _text_content(role: str, text: str):
    from google.genai import types

    return types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])


def _model_turn(res: LLMResult):
    from google.genai import types

    if res.raw_parts is not None:
        return res.raw_parts  # keep provider parts verbatim (thought signatures etc.)
    parts = [types.Part(text=res.text)] if res.text else []
    parts += [types.Part.from_function_call(name=c.name, args=c.args) for c in res.function_calls]
    return types.Content(role="model", parts=parts)


async def run_agent(ctx: AgentContext, user_message: str, *, max_steps: int | None = None) -> AgentResult:
    s = get_settings()
    max_steps = max_steps or s.agent_max_steps
    run_id = uuid.uuid4().hex[:16]
    result = AgentResult(reply="", run_id=run_id)

    history, saved_ids = await load_memory(ctx)
    system = f"{SYSTEM}\n\n--- Context ---\n{_context_block(ctx, saved_ids)}"
    contents = [_text_content(r, t) for r, t in history] + [_text_content("user", user_message)]
    tools = gemini_tools()
    seen: set[str] = set()
    n = 0

    for step in range(1, max_steps + 1):
        last = step == max_steps
        try:
            res = await ctx.llm.generate(contents, system=system, tools=None if last else tools)
        except Exception as exc:  # noqa: BLE001
            log.exception("llm failed in agent loop")
            result.steps.append(StepLog(step, "llm", ok=False, error=type(exc).__name__))
            result.reply = ("I'm having trouble reaching the language model right now. "
                            "Please try again in a moment.")
            break
        result.steps.append(StepLog(
            step, "llm", ok=True, latency_ms=res.latency_ms, prompt_tokens=res.prompt_tokens,
            output_tokens=res.output_tokens, text=res.text or None,
        ))
        calls = [] if last else res.function_calls
        if not calls:
            result.reply = res.text.strip() or "I couldn't produce an answer for that."
            result.hit_step_limit = last and bool(res.function_calls)
            break

        from google.genai import types

        contents.append(_model_turn(res))
        parts = []
        for call in calls:
            obs, tlog = await execute_tool(ctx, call, s.tool_timeout_s)
            tlog.step = step
            hint = check_observation(call.name, call.args, obs, seen)
            if hint:
                obs = {**obs, "note": hint}
            result.steps.append(tlog)
            parts.append(types.Part.from_function_response(name=call.name, response=obs))
        contents.append(types.Content(role="user", parts=parts))
        n += 1
    else:  # pragma: no cover - loop always breaks on `last`
        result.hit_step_limit = True

    await _persist(ctx, user_message, result)
    return result


async def _persist(ctx: AgentContext, user_message: str, result: AgentResult) -> None:
    sess = ctx.session
    try:
        sess.add(Message(user_id=ctx.user_id, role="user", content=user_message))
        sess.add(Message(user_id=ctx.user_id, role="assistant", content=result.reply,
                         meta={"state": ctx.state, "run_id": result.run_id}))
        for st in result.steps:
            sess.add(AgentStep(
                user_id=ctx.user_id, run_id=result.run_id, step=st.step, kind=st.kind, name=st.name,
                args=st.args, ok=st.ok, error=st.error, latency_ms=st.latency_ms,
                prompt_tokens=st.prompt_tokens, output_tokens=st.output_tokens,
            ))
        await sess.commit()
    except Exception:  # noqa: BLE001 - never lose the answer because logging failed
        log.exception("failed to persist agent run")
        await sess.rollback()
