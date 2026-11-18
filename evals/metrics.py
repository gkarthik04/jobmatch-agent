"""Ranking metrics with binary relevance. Pure functions (unit-tested)."""

import math


def recall_at_k(ranked: list, relevant: set, k: int = 10) -> float:
    """Capped recall: hits / min(|relevant|, k), so queries with >k relevant docs can still score 1."""
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / min(len(relevant), k)


def reciprocal_rank(ranked: list, relevant: set) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
