"""Retrieval eval: vector-only vs full-text-only vs hybrid (RRF) vs hybrid + LLM rerank.

Metrics (binary relevance, k=10): capped recall@10, MRR, nDCG@10, plus mean latency per query.
Runs against the frozen snapshot (id range in golden.json) so that postings added later by the
incremental ingestion job cannot change the numbers.
"""

import json
import time
from pathlib import Path

from app.db import get_sessionmaker
from app.services.search import SearchFilters, search
from evals.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

GOLDEN = Path(__file__).with_name("golden.json")
MODES = ["vector", "fts", "hybrid", "hybrid_rerank"]


def load_golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


async def run(embedder, llm, modes=None, k: int = 10, limit_cases: int | None = None, **search_kw) -> dict:
    g = load_golden()
    snap = g["snapshot"]
    filters = SearchFilters(id_range=(snap["id_min"], snap["id_max"]))
    cases = g["cases"][:limit_cases] if limit_cases else g["cases"]
    modes = modes or MODES
    out: dict = {"n_cases": len(cases), "snapshot": snap, "modes": {}, "per_case": {}}
    for mode in modes:
        recs, per_case = [], {}
        for c in cases:
            rel = set(c["relevant_source_ids"])
            t0 = time.perf_counter()
            async with get_sessionmaker()() as session:
                hits = await search(session, embedder, c["query"], filters=filters, mode=mode,
                                    limit=k, llm=llm, **search_kw)
            ms = (time.perf_counter() - t0) * 1000
            ranked = [h.job.source_id for h in hits]
            m = {"recall@10": recall_at_k(ranked, rel, k), "mrr": reciprocal_rank(ranked, rel),
                 "ndcg@10": ndcg_at_k(ranked, rel, k), "ms": ms}
            recs.append(m)
            per_case[c["id"]] = {**m, "query": c["query"]}
        out["modes"][mode] = {key: mean([r[key] for r in recs]) for key in ("recall@10", "mrr", "ndcg@10", "ms")}
        out["per_case"][mode] = per_case
    return out
