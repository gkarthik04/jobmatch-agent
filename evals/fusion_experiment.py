"""Offline fusion experiments: cache each retriever's ranking once, then try fusion variants.

Protocol (to limit overfitting to the golden set): queries are split into DEV (odd ids) and
HOLDOUT (even ids). Variants are chosen on DEV only; HOLDOUT numbers are what we report.
All variants tried are written to evals/results/retrieval_improvement.md.

    python -m evals.fusion_experiment
"""

import asyncio

from sqlalchemy import text

from app.db import dispose_engine, get_sessionmaker
from app.services.embeddings import QUERY, embed_cached, make_embedder
from app.services.search import SearchFilters, fts_ids, title_ids, vector_ids
from evals.metrics import mean, ndcg_at_k, reciprocal_rank
from evals.retrieval_eval import load_golden
from evals.run import RES

POOL = 100


def fuse(rankings: list[tuple[list[int], float]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ids, w in rankings:
        for r, i in enumerate(ids, start=1):
            scores[i] = scores.get(i, 0.0) + w / (k + r)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


VARIANTS = {  # name -> weights for (vector, fts, title_fts)
    "vector only": (1, 0, 0),
    "RRF vector+fts (baseline hybrid)": (1, 1, 0),
    "weighted RRF 1 : 0.5 (vec : fts)": (1, 0.5, 0),
    "weighted RRF 1 : 0.25 (vec : fts)": (1, 0.25, 0),
    "vector + title-fts": (1, 0, 1),
    "vector + title-fts (0.5)": (1, 0, 0.5),
    "vector + fts + title-fts (1 : 0.5 : 0.5)": (1, 0.5, 0.5),
    "vector + fts + title-fts (1 : 0.25 : 0.5)": (1, 0.25, 0.5),
}


async def main() -> None:
    g = load_golden()
    snap = g["snapshot"]
    f = SearchFilters(id_range=(snap["id_min"], snap["id_max"]))
    emb = make_embedder()
    cached = {}
    async with get_sessionmaker()() as s:
        id_by_src = dict((await s.execute(text("SELECT source_id, id FROM jobs"))).all())
        for c in g["cases"]:
            q = (await embed_cached(s, emb, [c["query"]], QUERY))[0]
            cached[c["id"]] = (await vector_ids(s, q, f, POOL), await fts_ids(s, c["query"], f, POOL),
                               await title_ids(s, c["query"], f, POOL))
    rel = {c["id"]: {id_by_src[x] for x in c["relevant_source_ids"]} for c in g["cases"]}
    splits = {"dev": [c["id"] for c in g["cases"] if c["id"] % 2 == 1],
              "holdout": [c["id"] for c in g["cases"] if c["id"] % 2 == 0]}
    lines = ["### Fusion experiment (all variants tried)\n",
             "Chosen on DEV (odd query ids) only; HOLDOUT (even ids) is the unbiased check.\n",
             "| Variant | DEV nDCG@10 | DEV MRR | HOLDOUT nDCG@10 | HOLDOUT MRR | ALL nDCG@10 |", "|---|---|---|---|---|---|"]
    for name, (wv, wf, wt) in VARIANTS.items():
        res = {}
        for split, ids in list(splits.items()) + [("all", [c["id"] for c in g["cases"]])]:
            nd, mr = [], []
            for cid in ids:
                v, fts, t = cached[cid]
                ranked = fuse([(v, wv), (fts, wf), (t, wt)])[:10]
                nd.append(ndcg_at_k(ranked, rel[cid], 10))
                mr.append(reciprocal_rank(ranked, rel[cid]))
            res[split] = (mean(nd), mean(mr))
        lines.append(f"| {name} | {res['dev'][0]:.3f} | {res['dev'][1]:.3f} | {res['holdout'][0]:.3f} | "
                     f"{res['holdout'][1]:.3f} | {res['all'][0]:.3f} |")
    RES.mkdir(exist_ok=True)
    (RES / "fusion_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
