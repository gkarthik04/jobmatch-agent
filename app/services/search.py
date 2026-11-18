"""Hybrid retrieval: Postgres full-text (ts_rank_cd) + pgvector cosine, fused with RRF."""

import re
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Job
from app.services.embeddings import QUERY, Embedder, embed_cached

Mode = Literal["vector", "fts", "hybrid", "hybrid_rerank"]
POOL = 100  # candidates pulled from each retriever before fusion


@dataclass
class SearchFilters:
    location: str | None = None
    remote: bool | None = None
    posted_within_days: int | None = None  # relative to the newest posting in the index
    exclude_ids: list[int] = field(default_factory=list)


@dataclass
class Hit:
    job: Job
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    rerank_score: float | None = None
    rerank_reason: str | None = None


_STOP = set(
    "a an and are as at be by for from has have in is it of on or that the this to with we you your "
    "our will can job role team work working experience years year strong ability skills".split()
)


def build_or_tsquery(text_: str, max_terms: int = 30) -> str:
    """Natural language / resume text -> 'a | b | c' tsquery (OR semantics, so long queries match)."""
    seen: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9+#.]*[A-Za-z0-9+#]|[A-Za-z]", text_.lower()):
        tok = re.sub(r"[^a-z0-9]", "", tok)  # strip punctuation: tsquery-safe
        if len(tok) > 1 and tok not in _STOP and tok not in seen:
            seen.append(tok)
    return " | ".join(seen[:max_terms])


def rrf_fuse(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)). Returns [(id, score)] desc."""
    scores: dict[int, float] = {}
    for ids in rankings.values():
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def build_filters(f: SearchFilters) -> tuple[str, dict]:
    clauses: list[str] = []
    params: dict = {}
    if f.location:
        clauses.append("location ILIKE :loc")
        params["loc"] = f"%{f.location}%"
    if f.remote is not None:
        clauses.append("remote = :remote")
        params["remote"] = f.remote
    if f.posted_within_days:
        clauses.append(
            "posted_at >= (SELECT max(posted_at) FROM jobs) - make_interval(days => :days)"
        )
        params["days"] = int(f.posted_within_days)
    if f.exclude_ids:
        clauses.append("id <> ALL(:excl)")
        params["excl"] = list(f.exclude_ids)
    return ("".join(f" AND {c}" for c in clauses), params)


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def vector_ids(session: AsyncSession, qvec: list[float], f: SearchFilters, n: int) -> list[int]:
    where, params = build_filters(f)
    await session.execute(text("SET LOCAL hnsw.ef_search = 100"))
    rows = await session.execute(
        text(
            f"SELECT id FROM jobs WHERE embedding IS NOT NULL{where} "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :n"
        ),
        {**params, "q": vec_literal(qvec), "n": n},
    )
    return [r[0] for r in rows]


async def fts_ids(session: AsyncSession, query: str, f: SearchFilters, n: int) -> list[int]:
    tsq = build_or_tsquery(query)
    if not tsq:
        return []
    where, params = build_filters(f)
    rows = await session.execute(
        text(
            "SELECT id FROM jobs, to_tsquery('english', :tsq) q "
            f"WHERE tsv @@ q{where} ORDER BY ts_rank_cd(tsv, q, 32) DESC, id LIMIT :n"
        ),
        {**params, "tsq": tsq, "n": n},
    )
    return [r[0] for r in rows]


async def search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    *,
    filters: SearchFilters | None = None,
    mode: Mode = "hybrid",
    limit: int = 10,
    qvec: list[float] | None = None,
    llm=None,
) -> list[Hit]:
    """Run retrieval in the requested mode. `qvec` lets callers (matching) supply their own vector."""
    f = filters or SearchFilters()
    s = get_settings()
    rankings: dict[str, list[int]] = {}

    if mode in ("vector", "hybrid", "hybrid_rerank"):
        if qvec is None:
            qvec = (await embed_cached(session, embedder, [query], QUERY))[0]
        rankings["vector"] = await vector_ids(session, qvec, f, POOL)
    if mode in ("fts", "hybrid", "hybrid_rerank"):
        rankings["fts"] = await fts_ids(session, query, f, POOL)

    fused = rrf_fuse(rankings, k=s.rrf_k)
    pool = fused[: s.rerank_top_n if mode == "hybrid_rerank" else limit]
    ids = [i for i, _ in pool]
    if not ids:
        return []
    jobs = {j.id: j for j in (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars()}
    hits = []
    for jid, score in pool:
        if jid in jobs:
            ranks = {name: lst.index(jid) + 1 for name, lst in rankings.items() if jid in lst}
            hits.append(Hit(job=jobs[jid], score=score, ranks=ranks))

    if mode == "hybrid_rerank":
        from app.services.rerank import rerank

        hits = await rerank(session, llm, query, hits)
    return hits[:limit]
