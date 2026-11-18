import numpy as np

from app.services.preferences import apply_preference, preference_delta, rocchio


def unit(v):
    v = np.asarray(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def rank(query, docs):
    """Rank doc names by cosine similarity (all vectors are unit length)."""
    return [n for n, v in sorted(docs.items(), key=lambda kv: -float(np.dot(query, kv[1])))]


# 3-d toy space: axis0 = backend, axis1 = data, axis2 = frontend
DOCS = {
    "backend_a": unit([1.0, 0.1, 0.0]),
    "backend_b": unit([0.9, 0.2, 0.1]),
    "data_a": unit([0.3, 1.0, 0.0]),
    "data_b": unit([0.2, 0.9, 0.1]),
    "frontend_a": unit([0.1, 0.0, 1.0]),
}
PROFILE_Q = unit([1.0, 0.8, 0.1])  # ambiguous profile: backend + data


def test_no_feedback_leaves_query_unchanged():
    assert np.allclose(rocchio(PROFILE_Q, [], []), PROFILE_Q)
    assert preference_delta([], []) is None


def test_liking_data_jobs_moves_data_jobs_up_before_after():
    before = rank(PROFILE_Q, DOCS)
    after_q = rocchio(PROFILE_Q, liked=[DOCS["data_a"], DOCS["data_b"]], disliked=[])
    after = rank(after_q, DOCS)
    print("before:", before, "\nafter: ", after)
    assert before != after
    assert before.index("data_a") > after.index("data_a") or before[0] == "data_a"
    assert after[0].startswith("data")
    assert after.index("data_b") < before.index("data_b") or before.index("data_b") == 0


def test_disliking_pushes_similar_jobs_down():
    before = rank(PROFILE_Q, DOCS)
    after_q = rocchio(PROFILE_Q, liked=[], disliked=[DOCS["backend_a"], DOCS["backend_b"]])
    after = rank(after_q, DOCS)
    assert after.index("backend_a") > before.index("backend_a")
    assert after.index("backend_b") > before.index("backend_b")


def test_delta_is_recomputed_not_accumulated():
    d1 = preference_delta([DOCS["data_a"]], [])
    d2 = preference_delta([DOCS["data_a"]], [])
    assert d1 == d2  # idempotent: same feedback set -> same vector
    assert preference_delta([], []) is None  # un-liking everything clears the preference


def test_result_is_unit_length():
    q = apply_preference(PROFILE_Q, preference_delta([DOCS["data_a"]], [DOCS["frontend_a"]]))
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
