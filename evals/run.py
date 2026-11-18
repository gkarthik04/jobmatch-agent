"""One command for all evals:  python -m evals.run  [--only retrieval|structured|agent|judge] [--tag NAME]

Raw results go to evals/results/<suite>_<tag>.json; evals/RESULTS.md is re-rendered from every
result file present, so partial / repeated runs accumulate into one report. Nothing here is
hand-edited: every number in RESULTS.md was produced by these scripts.
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.db import dispose_engine
from app.logging_conf import setup_logging
from app.services.embeddings import make_embedder
from app.services.llm import get_llm
from evals.metrics import md_table

HERE = Path(__file__).parent
RES = HERE / "results"


def save(suite: str, tag: str, data: dict) -> None:
    RES.mkdir(exist_ok=True)
    data["_meta"] = {"suite": suite, "tag": tag, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    (RES / f"{suite}_{tag}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def load(suite: str, tag: str) -> dict | None:
    f = RES / f"{suite}_{tag}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


# ---- rendering --------------------------------------------------------------------------------
def render_retrieval(out: list[str]) -> None:
    cur, base = load("retrieval", "current"), load("retrieval", "baseline")
    if not cur and not base:
        return
    out.append("## 1. Retrieval quality\n")
    ref = cur or base
    out.append(
        f"{ref['n_cases']} golden queries over a frozen snapshot of {ref['snapshot']['n_jobs']:,} postings "
        "(labels from explicit title rubrics in `evals/golden_queries.py`, not from any retriever). "
        "recall@10 is capped: hits / min(#relevant, 10).\n")
    names = {"vector": "Vector only", "fts": "Full-text only (ts_rank_cd)", "hybrid": "Hybrid (RRF)",
             "hybrid_rerank": "Hybrid + LLM rerank (top 50)"}
    for tag, data, title in (("baseline", base, "Baseline"), ("current", cur, "After improvement")):
        if not data or (tag == "current" and base and data is base):
            continue
        rows = [[names.get(m, m), v["recall@10"], v["mrr"], v["ndcg@10"], f"{v['ms']:.0f} ms"]
                for m, v in data["modes"].items()]
        out.append(f"**{title}** (run {data['_meta']['at']})\n")
        out.append(md_table(["Retriever", "recall@10", "MRR", "nDCG@10", "mean latency"], rows) + "\n")
    if base and cur:
        rows = []
        for m in cur["modes"]:
            if m in base["modes"]:
                b, c = base["modes"][m], cur["modes"][m]
                rows.append([names.get(m, m), f"{b['ndcg@10']:.3f} → {c['ndcg@10']:.3f}",
                             f"{b['mrr']:.3f} → {c['mrr']:.3f}", f"{b['recall@10']:.3f} → {c['recall@10']:.3f}"])
        out.append("**Before → after**\n")
        out.append(md_table(["Retriever", "nDCG@10", "MRR", "recall@10"], rows) + "\n")
    note = (RES / "retrieval_improvement.md")
    if note.exists():
        out.append(note.read_text(encoding="utf-8") + "\n")


def render_structured(out: list[str]) -> None:
    d = load("structured", "current")
    if not d:
        return
    out.append("## 2. Structured output (resume PDF → validated profile)\n")
    v1, v2 = d.get("v1"), d.get("v2")
    out.append(f"{v2['n']} synthetic resumes with known ground truth (6 tracks × 3 seniority levels, "
               "varied layouts; PDFs → text → Gemini JSON-schema output → Pydantic validation).\n")
    rows = [["Schema-valid on first attempt", f"{v1['valid_first_try']:.1%}", f"{v2['valid_first_try']:.1%}"],
            ["Schema-valid after validate→retry", f"{v1['valid_after_retry']:.1%}", f"{v2['valid_after_retry']:.1%}"]]
    for k in v2["fields"]:
        a, b = v1["fields"].get(k, (None, 0)), v2["fields"][k]
        rows.append([k, "-" if a[0] is None else f"{a[0]:.3f}", f"{b[0]:.3f}" + (f" (n={b[1]})" if b[1] != v2["n"] else "")])
    out.append(md_table(["Metric", "Prompt v1 (baseline)", "Prompt v2 (current)"], rows) + "\n")


def render_agent(out: list[str]) -> None:
    d = load("agent", "current")
    if not d:
        return
    out.append("## 3. Agent (tool use, memory, step limit)\n")
    out.append(f"{d['n_conversations']} scripted multi-turn conversations ({d['n_turns']} turns) against the real "
               "agent, tools and database. Follow-ups (\"the 2nd one\", \"only remote\") must resolve from "
               "Postgres-backed memory.\n")
    rows = [["Conversations fully passed", f"{d['conversations_passed']:.1%}"],
            ["Turns with the right tool(s), in order", f"{d['tool_selection']:.1%}"],
            ["Turns with the right arguments (incl. resolved job ids)", f"{d['argument_accuracy']:.1%}"],
            ["Turns completed (non-empty answer, no step-limit / outage)", f"{d['task_completion']:.1%}"],
            ["Turns within the 6-step limit", f"{d['within_step_limit']:.1%}"],
            ["Mean LLM calls per turn", f"{d['mean_llm_calls_per_turn']:.2f}"],
            ["Mean latency per turn", f"{d['mean_latency_ms'] / 1000:.1f} s"],
            ["Mean tokens per turn", f"{d['mean_tokens_per_turn']:.0f}"]]
    out.append(md_table(["Metric", "Result"], rows) + "\n")
    rows = [[c["name"], "pass" if c["passed"] else "FAIL", " | ".join(",".join(t["tools_called"]) or "-" for t in c["turns"])]
            for c in d["conversations"]]
    out.append(md_table(["Conversation", "Result", "Tools called (per turn)"], rows) + "\n")


def render_judge(out: list[str]) -> None:
    d = load("judge", "current")
    if not d:
        return
    out.append("## 4. Match-explanation quality (LLM-as-judge)\n")
    out.append(f"{d['n_explanations']} explanations for 6 synthetic personas, graded 1-5 by a hand-written judge "
               "prompt against only the posting + profile text. Caveat: judge and generator are the same model "
               "family, so scores are optimistic; treat as a regression signal, not absolute truth.\n")
    rows = [["Groundedness (mean, 1-5)", f"{d['groundedness_mean']:.2f}"],
            ["Relevance (mean, 1-5)", f"{d['relevance_mean']:.2f}"],
            ["Explanations with groundedness ≥ 4", f"{d['grounded_ge4']:.1%}"],
            ["Explanations with relevance ≥ 4", f"{d['relevant_ge4']:.1%}"],
            ["Explanations flagged with unsupported claims", f"{d['with_unsupported_claims']:.1%}"]]
    out.append(md_table(["Metric", "Result"], rows) + "\n")


def render() -> None:
    out = ["# Evaluation results\n",
           "_Generated by `python -m evals.run`. Every number below comes from the result files in "
           "`evals/results/`; none are hand-entered._\n"]
    for fn in (render_retrieval, render_structured, render_agent, render_judge):
        fn(out)
    (HERE / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")


# ---- runners ----------------------------------------------------------------------------------
async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["retrieval", "structured", "agent", "judge"])
    ap.add_argument("--tag", default="current")
    ap.add_argument("--modes", help="comma list of retrieval modes")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    if a.render_only:
        render()
        return
    setup_logging("WARNING")
    embedder, llm = make_embedder(), get_llm()
    suites = [a.only] if a.only else ["retrieval", "structured", "agent", "judge"]
    for suite in suites:
        t0 = time.time()
        print(f"== {suite} ==", flush=True)
        if suite == "retrieval":
            from evals import retrieval_eval

            modes = a.modes.split(",") if a.modes else None
            data = await retrieval_eval.run(embedder, llm, modes=modes)
        elif suite == "structured":
            from app.services.resume import PROMPTS
            from evals import structured_eval

            data = {v: await structured_eval.run(llm, system=PROMPTS[v]) for v in ("v1", "v2")}
        elif suite == "agent":
            from evals import agent_eval

            data = await agent_eval.run(embedder, llm)
        else:
            from evals import judge_eval

            data = await judge_eval.run(embedder, llm)
        save(suite, a.tag, data)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
        render()
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
