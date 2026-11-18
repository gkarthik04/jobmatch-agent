"""Rocchio-style relevance feedback in embedding space.

    q_new = normalize( alpha * q0  +  beta * mean(liked)  -  gamma * mean(disliked) )

q0 is the profile query vector. We store only the feedback term (`beta*mean(liked) - gamma*mean(disliked)`)
per user and recompute it from all feedback rows on every like/dislike, so it never drifts and
re-liking/un-liking a job is exact. The final query vector is built at match time.
"""

import numpy as np

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.25


def _mean(vecs: list[list[float]]) -> np.ndarray | None:
    return np.mean(np.asarray(vecs, dtype=float), axis=0) if vecs else None


def preference_delta(
    liked: list[list[float]], disliked: list[list[float]], beta: float = BETA, gamma: float = GAMMA
) -> list[float] | None:
    pos, neg = _mean(liked), _mean(disliked)
    if pos is None and neg is None:
        return None
    delta = np.zeros_like(pos if pos is not None else neg)
    if pos is not None:
        delta = delta + beta * pos
    if neg is not None:
        delta = delta - gamma * neg
    return delta.tolist()


def apply_preference(q0: list[float], delta: list[float] | None, alpha: float = ALPHA) -> list[float]:
    q = alpha * np.asarray(q0, dtype=float)
    if delta is not None:
        q = q + np.asarray(delta, dtype=float)
    n = np.linalg.norm(q) or 1.0
    return (q / n).tolist()


def rocchio(q0, liked, disliked, alpha=ALPHA, beta=BETA, gamma=GAMMA) -> list[float]:
    return apply_preference(q0, preference_delta(liked, disliked, beta, gamma), alpha)
