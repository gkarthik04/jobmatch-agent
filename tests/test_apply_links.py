from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.services.agent_tools import AgentContext, JobIdArgs, get_job_details
from app.services.ingest import clean_row, normalize_link
from app.services.job_status import OPEN_SQL, is_open, link_for, source_label
from app.services.live_feed import clean_arbeitnow, clean_feed, looks_english, make_source_id
from app.services.search import SearchFilters, build_filters

DESC = "We are looking for an engineer to build data pipelines with Python and SQL. " * 5
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


# ---- link safety -------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "javascript:alert(1)", "data:text/html;base64,AAAA", "mailto:a@b.com", "ftp://x.org/file",
    "//evil.example", "file:///etc/passwd", "nan", "", None, "not a url", "https://" + "a" * 1100,
])
def test_unsafe_or_junk_links_are_rejected(bad):
    assert normalize_link(bad) is None


def test_good_links_are_kept_and_tracking_stripped():
    assert normalize_link("www.childrensnebraska.org") == "https://www.childrensnebraska.org"
    assert normalize_link("https://x.org/jobs/1?trk=abc&id=7", drop_params=("trk",)) == "https://x.org/jobs/1?id=7"
    assert normalize_link("http://x.org/a") == "http://x.org/a"


def test_clean_row_carries_links_and_expiry():
    row = {"job_id": 9.0, "title": "Data Engineer", "company_name": "Acme", "description": DESC,
           "location": "Austin, TX", "listed_time": 1712345678000.0,
           "job_posting_url": "https://www.linkedin.com/jobs/view/9/?trk=jobs_biz",
           "application_url": "careers.acme.com/apply", "expiry": 1715985908000.0}
    j = clean_row(row)
    assert j.url == "https://www.linkedin.com/jobs/view/9/"
    assert j.apply_url == "https://careers.acme.com/apply"
    assert j.source == "linkedin_2024" and j.expires_at.year == 2024 and j.expires_at.month == 5
    bad = clean_row({**row, "job_posting_url": "javascript:alert(1)", "application_url": float("nan")})
    assert bad.url is None and bad.apply_url is None


# ---- what counts as open -----------------------------------------------------------------------
def job(**kw):
    base = dict(source="arbeitnow", last_seen_at=NOW - timedelta(days=1), expires_at=None,
                url="https://a.example/j", apply_url=None)
    return SimpleNamespace(**{**base, **kw})


def test_open_rules():
    assert is_open(job(), NOW)
    assert not is_open(job(last_seen_at=NOW - timedelta(days=5)), NOW)  # fell out of the feed
    assert not is_open(job(last_seen_at=None), NOW)
    assert not is_open(job(expires_at=NOW - timedelta(days=1)), NOW)  # past an explicit expiry
    assert not is_open(job(source="linkedin_2024", last_seen_at=NOW), NOW)  # archive is never open


def test_link_prefers_apply_url_and_labels():
    assert link_for(job(apply_url="https://apply.example/x")) == "https://apply.example/x"
    assert link_for(job()) == "https://a.example/j"
    assert source_label("arbeitnow") == "Arbeitnow" and source_label("linkedin_2024") == "LinkedIn, April 2024"


def test_open_filter_adds_sql_clause():
    where, _ = build_filters(SearchFilters(open_only=True))
    assert OPEN_SQL in where
    assert build_filters(SearchFilters()) == ("", {})


# ---- live feed cleaning ------------------------------------------------------------------------
RAW = {"slug": "data-engineer-berlin-1", "title": "Data Engineer", "company_name": "Beta GmbH",
       "description": "<p>" + DESC + "</p>", "remote": True, "url": "https://www.arbeitnow.com/jobs/companies/beta/x",
       "tags": ["Python"], "location": "Berlin", "created_at": 1795023048}


def test_arbeitnow_job_is_cleaned_with_apply_link():
    j = clean_arbeitnow(RAW, NOW)
    assert j.source_id == make_source_id("data-engineer-berlin-1") and j.source == "arbeitnow"
    assert len(make_source_id("x" * 300)) <= 64  # long slugs fit the column
    assert j.apply_url == j.url == RAW["url"] and j.last_seen_at == NOW and j.expires_at is None
    assert "python" in j.skills and "<p>" not in j.description and j.remote is True


def test_feed_skips_german_bad_links_and_duplicates():
    german = {**RAW, "slug": "de-1", "description": "Wir suchen einen Entwickler für unser Team und wir bieten "
              "Ihnen eine sichere Stelle mit einem tollen Gehalt und der Chance zu wachsen. " * 4}
    unsafe = {**RAW, "slug": "bad-1", "url": "javascript:alert(1)"}
    out = clean_feed([RAW, RAW, german, unsafe], NOW)
    assert [j.source_id for j in out] == [make_source_id("data-engineer-berlin-1")]
    assert looks_english(DESC) and not looks_english(german["description"])


# ---- agent ------------------------------------------------------------------------------------
class FakeSession:
    def __init__(self, j):
        self.j = j

    async def get(self, model, id):
        return self.j


def full_job(**kw):
    base = dict(id=5, title="T", company="C", location="L", remote=False, skills=["python"],
                experience_level=None, posted_at=None, description="d" * 50, source="arbeitnow",
                url="https://a.example/j", apply_url="https://a.example/apply",
                expires_at=None, last_seen_at=datetime.now(UTC))
    return SimpleNamespace(**{**base, **kw})


async def test_agent_details_give_apply_link_and_flag_archived_listings():
    ctx = AgentContext(session=FakeSession(full_job()), user_id=1, embedder=None, llm=None)
    live = await get_job_details(ctx, JobIdArgs(job_id=5))
    assert live["apply_link"] == "https://a.example/apply" and live["apply_note"] is None
    old = full_job(source="linkedin_2024", last_seen_at=None, expires_at=datetime(2024, 5, 1, tzinfo=UTC))
    ctx = AgentContext(session=FakeSession(old), user_id=1, embedder=None, llm=None)
    archived = await get_job_details(ctx, JobIdArgs(job_id=5))
    assert "probably closed" in archived["apply_note"] and archived["source"] == "LinkedIn, April 2024"


def test_feed_text_cleanup_removes_broken_dash_and_promo_footer():
    raw = {**RAW, "title": "Working Student AI Engineer \ufffd LLMs",
           "description": DESC + " Find more English Speaking Jobs in Germany on Arbeitnow"}
    j = clean_arbeitnow(raw, NOW)
    assert "\ufffd" not in j.title and "Arbeitnow" not in j.description
