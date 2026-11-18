import math

from evals.metrics import md_table, mean, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_is_capped_by_k():
    rel = set(range(100))
    assert recall_at_k(list(range(10)), rel, 10) == 1.0  # 10 hits, only 10 slots
    assert recall_at_k([200, 201, 0], {0, 1}, 10) == 0.5
    assert recall_at_k([1], set(), 10) == 0.0


def test_mrr():
    assert reciprocal_rank([9, 8, 1], {1}) == 1 / 3
    assert reciprocal_rank([9, 8], {1}) == 0.0


def test_ndcg_perfect_and_worse():
    rel = {1, 2}
    assert math.isclose(ndcg_at_k([1, 2, 3], rel), 1.0)
    worse = ndcg_at_k([3, 1, 2], rel)
    assert 0 < worse < 1
    assert math.isclose(worse, (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


def test_mean_and_table():
    assert mean([]) == 0.0 and mean([1, 3]) == 2
    t = md_table(["a", "b"], [["x", 0.12345]])
    assert "| x | 0.123 |" in t
