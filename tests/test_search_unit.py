from types import SimpleNamespace

from app.services.rerank import apply_scores
from app.services.search import SearchFilters, build_filters, build_or_tsquery, rrf_fuse, vec_literal


def test_rrf_rewards_agreement_between_retrievers():
    fused = rrf_fuse({"vector": [1, 2, 3, 4], "fts": [3, 1, 9, 8]}, k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == [1, 3]  # both retrievers rank 1 and 3 highly
    assert set(ids) == {1, 2, 3, 4, 8, 9}


def test_rrf_scores_are_exact():
    (top_id, top_score), *_ = rrf_fuse({"a": [7], "b": [7]}, k=60)
    assert top_id == 7 and abs(top_score - 2 / 61) < 1e-12


def test_rrf_single_ranker_preserves_order():
    assert [i for i, _ in rrf_fuse({"a": [5, 4, 3]})] == [5, 4, 3]


def test_or_tsquery_is_safe_and_deduped():
    q = build_or_tsquery("Senior Python/SQL engineer; python, C++ & 'drop table'!")
    terms = q.split(" | ")
    assert terms.count("python") == 1
    assert all(t.isalnum() for t in terms), terms
    assert "senior" in terms


def test_or_tsquery_empty_for_stopwords_only():
    assert build_or_tsquery("the and of") == ""


def test_filters_sql_and_params():
    where, params = build_filters(SearchFilters(location="Austin", remote=True, posted_within_days=7))
    assert "location ILIKE :loc" in where and "remote = :remote" in where and ":days" in where
    assert params["loc"] == "%Austin%" and params["remote"] is True and params["days"] == 7
    assert build_filters(SearchFilters()) == ("", {})


def test_vec_literal_format():
    assert vec_literal([0.5, -1.0]) == "[0.500000,-1.000000]"


def _hit(i):
    return SimpleNamespace(job=SimpleNamespace(id=i), rerank_score=None)


def test_apply_scores_orders_and_keeps_rrf_for_ties_and_missing():
    hits = [_hit(1), _hit(2), _hit(3), _hit(4)]
    out = apply_scores(hits, {2: 9.0, 3: 5.0, 4: 5.0})  # 1 unscored
    assert [h.job.id for h in out] == [2, 3, 4, 1]


def test_weighted_rrf_downweights_a_noisy_retriever():
    rankings = {"vector": [1, 2, 3], "fts": [3, 2, 1]}
    plain = [i for i, _ in rrf_fuse(rankings)]
    weighted = [i for i, _ in rrf_fuse(rankings, weights={"vector": 1.0, "fts": 0.25})]
    assert weighted[0] == 1  # vector's favourite wins once fts is down-weighted
    assert plain != weighted or plain[0] == 1
    (_, s1), = rrf_fuse({"a": [9]}, weights={"a": 0.5})
    assert abs(s1 - 0.5 / 61) < 1e-12


def test_default_weight_map_parses():
    from app.config import Settings

    assert Settings(rrf_weights="1,0.5,0.5").rrf_weight_map == {"vector": 1.0, "fts": 0.5, "title": 0.5}


async def test_rerank_times_out_and_falls_back_to_rrf_order(monkeypatch):
    import asyncio

    from app.config import get_settings
    from app.services import rerank as rr

    async def hang(session, key, produce):
        await asyncio.sleep(5)

    monkeypatch.setattr(rr, "cached_json", hang)
    monkeypatch.setattr(get_settings(), "rerank_budget_s", 0.05)
    hits = [_hit(1), _hit(2), _hit(3)]
    out = await rr.rerank(None, object(), "q", hits)
    assert [h.job.id for h in out] == [1, 2, 3]  # untouched RRF order, no exception


def test_dedupe_hits_collapses_same_title_and_company_keeping_best():
    from app.services.search import dedupe_hits

    def h(i, title, company):
        return SimpleNamespace(job=SimpleNamespace(id=i, title=title, company=company))

    hits = [h(1, "Frontend Dev", "Acme"), h(2, "frontend dev ", "ACME"), h(3, "Frontend Dev", "Beta"),
            h(4, "Backend Dev", "Acme")]
    out = dedupe_hits(hits)
    assert [x.job.id for x in out] == [1, 3, 4]
    assert [x.job.id for x in dedupe_hits(hits, limit=2)] == [1, 3]
