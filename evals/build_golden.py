"""Apply the rubrics in golden_queries.py to the frozen index snapshot -> evals/golden.json.

    python -m evals.build_golden
"""

import asyncio
import json
import re
from pathlib import Path

from sqlalchemy import text

from app.db import dispose_engine, get_sessionmaker
from evals.golden_queries import QUERIES

OUT = Path(__file__).with_name("golden.json")
MIN_REL, MAX_REL = 3, 400


async def main() -> None:
    async with get_sessionmaker()() as s:
        rows = (await s.execute(text(
            "SELECT id, source_id, title FROM jobs WHERE source_id NOT LIKE 'test-%' ORDER BY id"))).all()
    ids = [r[0] for r in rows]
    snapshot = {"id_min": min(ids), "id_max": max(ids), "n_jobs": len(ids)}
    cases, dropped = [], []
    for q, must, must_not in QUERIES:
        rx, nx = re.compile(must, re.I), re.compile(must_not, re.I) if must_not else None
        rel = [r[1] for r in rows if rx.search(r[2]) and not (nx and nx.search(r[2]))]
        (cases if MIN_REL <= len(rel) <= MAX_REL else dropped).append(
            {"id": len(cases) + 1, "query": q, "rubric": {"title_matches": must, "title_not_matches": must_not},
             "n_relevant": len(rel), "relevant_source_ids": rel} if MIN_REL <= len(rel) <= MAX_REL
            else (q, len(rel)))
    OUT.write_text(json.dumps({"snapshot": snapshot, "cases": cases}, indent=1))
    print(f"snapshot={snapshot}; wrote {len(cases)} cases to {OUT.name}")
    for c in cases:
        print(f"  {c['n_relevant']:4d}  {c['query']}")
    for q, n in dropped:
        print(f"  DROPPED ({n} relevant): {q}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
