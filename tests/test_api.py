import pytest

from app.ratelimit import SlidingWindow
from app.schemas import ProfileSchema
from app.security import create_token, decode_token, hash_password, verify_password
from app.services.llm import FunctionCall, LLMResult
from tests.conftest import make_pdf

RESUME = [
    "Test Candidate", "B.Tech Computer Science, XYZ University, 2025",
    "Skills: Python, SQL, Docker, FAISS", "Projects: RAG bot using FAISS and Sentence Transformers",
    "Experience: 6 month backend internship building REST APIs with Flask and MySQL",
]


# ---- pure unit tests (no DB) ------------------------------------------------------------------
def test_password_hash_roundtrip():
    h = hash_password("s3cretpass")
    assert h != "s3cretpass" and verify_password("s3cretpass", h) and not verify_password("nope", h)


def test_token_roundtrip_and_tamper():
    tok = create_token(42)
    assert decode_token(tok) == 42
    from app.errors import AppError

    with pytest.raises(AppError):
        decode_token(tok[:-2] + "xx")


def test_sliding_window_limits_and_recovers():
    w = SlidingWindow(2, 60)
    assert w.check("u", now=0) == 0 and w.check("u", now=1) == 0
    assert w.check("u", now=2) > 0  # third call within the window is rejected
    assert w.check("other", now=2) == 0  # per-key isolation
    assert w.check("u", now=61) == 0  # window slid


def test_profile_schema_rejects_bad_types():
    with pytest.raises(ValueError):
        ProfileSchema.model_validate({"name": "x", "skills": "python"})


# ---- DB-backed API tests ----------------------------------------------------------------------
@pytest.mark.db
async def test_auth_flow(client):
    creds = {"email": "test-auth1@example.com", "password": "longenough1"}
    assert (await client.post("/auth/signup", json=creds)).status_code == 201
    dup = await client.post("/auth/signup", json=creds)
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "email_taken"
    ok = await client.post("/auth/login", json=creds)
    assert ok.status_code == 200
    h = {"Authorization": f"Bearer {ok.json()['access_token']}"}
    assert (await client.get("/auth/me", headers=h)).json()["email"] == creds["email"]
    bad = await client.post("/auth/login", json={**creds, "password": "wrongwrong"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "bad_credentials"
    assert (await client.post("/auth/signup", json={"email": "test-weak@example.com", "password": "x"})
            ).status_code == 422
    assert (await client.get("/auth/me")).status_code == 401


@pytest.mark.db
async def test_resume_upload_and_isolation(client, auth):
    r = await client.post("/resume", headers=auth, files={"file": ("cv.pdf", make_pdf(RESUME), "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Test Candidate" and r.json()["skills"].count("Python") == 1  # deduped
    assert (await client.get("/profile", headers=auth)).status_code == 200
    # another user must not see it
    other = await client.post("/auth/signup", json={"email": "test-other@example.com", "password": "longenough1"})
    h2 = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert (await client.get("/profile", headers=h2)).status_code == 404
    bad = await client.post("/resume", headers=auth, files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "not_a_pdf"


@pytest.mark.db
async def test_search_hybrid_and_filters(client, seed_jobs):
    r = await client.get("/search", params={"q": "zyxwvu backend python", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    top = body["results"][0]
    assert top["title"] == "Zyxwvu Backend Engineer"
    assert set(top["ranks"]) == {"vector", "fts"}  # both retrievers contributed
    rem = (await client.get("/search", params={"q": "zyxwvu", "remote": True})).json()["results"]
    assert rem and all(x["remote"] for x in rem)
    loc = (await client.get("/search", params={"q": "zyxwvu", "location": "austin"})).json()["results"]
    assert {x["company"] for x in loc if x["title"].startswith("Zyxwvu")} == {"Acme", "Gamma"}
    for mode in ("vector", "fts", "hybrid_rerank"):
        assert (await client.get("/search", params={"q": "zyxwvu sql", "mode": mode})).status_code == 200
    assert (await client.get("/search", params={"q": "x"})).status_code == 422


@pytest.mark.db
async def test_matches_feedback_changes_ranking(client, auth, seed_jobs):
    await client.post("/resume", headers=auth, files={"file": ("cv.pdf", make_pdf(RESUME), "application/pdf")})
    m = await client.get("/matches", headers=auth, params={"limit": 10})
    assert m.status_code == 200
    items = [x for x in m.json() if x["job"]["title"].startswith("Zyxwvu")]
    assert items and items[0]["why_it_matches"] and "fit_score" in items[0]
    assert isinstance(items[0]["missing_skills"], list)
    first = items[0]["job"]["id"]
    # dislike the top match: it must disappear from future matches
    fb = await client.post("/feedback", headers=auth, json={"job_id": first, "liked": False})
    assert fb.status_code == 200 and fb.json()["feedback_count"] == 1
    after = await client.get("/matches", headers=auth, params={"limit": 20})
    assert first not in [x["job"]["id"] for x in after.json()]
    from app.models import Profile

    async with client.maker() as s:
        uid = (await client.get("/auth/me", headers=auth)).json()["id"]
        assert (await s.get(Profile, uid)).pref_vector is not None
    assert (await client.post("/feedback", headers=auth, json={"job_id": 10**9, "liked": True})
            ).status_code == 404


@pytest.mark.db
async def test_matches_requires_resume(client, auth):
    r = await client.get("/matches", headers=auth)
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_profile"


@pytest.mark.db
async def test_chat_agent_uses_tools_remembers_and_saves(client, auth, seed_jobs, fake_llm):
    # turn 1: search, then answer
    fake_llm.script = [
        LLMResult(text="Plan: search.", function_calls=[FunctionCall("search_jobs", {"query": "zyxwvu"})]),
        LLMResult(text="Found 3 jobs."),
    ]
    r1 = await client.post("/chat", headers=auth, json={"message": "find zyxwvu jobs"})
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert [s["name"] for s in d1["steps"] if s["kind"] == "tool"] == ["search_jobs"]
    assert d1["reply"] == "Found 3 jobs." and d1["usage"]["llm_calls"] == 2

    # turn 2: "save the 1st one" -> model must be told the previous results (memory in Postgres)
    seen_system = {}
    orig = fake_llm.generate

    async def spy(contents, **kw):
        seen_system["s"] = kw.get("system")
        return await orig(contents, **kw)

    fake_llm.generate = spy
    from sqlalchemy import select

    from app.models import Message

    async with client.maker() as s:
        uid = (await client.get("/auth/me", headers=auth)).json()["id"]
        meta = (await s.execute(select(Message.meta).where(
            Message.user_id == uid, Message.role == "assistant"))).scalars().first()
    first_id = meta["state"]["last_results"][0]
    fake_llm.script = [
        LLMResult(function_calls=[FunctionCall("save_job", {"job_id": first_id})]),
        LLMResult(text="Saved."),
    ]
    r2 = await client.post("/chat", headers=auth, json={"message": "save the first one"})
    assert r2.status_code == 200
    assert f"1->{first_id}" in seen_system["s"]  # follow-up context injected from DB memory
    saved = (await client.get("/saved", headers=auth)).json()
    assert [x["id"] for x in saved] == [first_id]
    hist = (await client.get("/chat/history", headers=auth)).json()
    assert [h["role"] for h in hist] == ["user", "assistant", "user", "assistant"]
    # isolation: a different user has no saved jobs and no history
    o = await client.post("/auth/signup", json={"email": "test-iso@example.com", "password": "longenough1"})
    h2 = {"Authorization": f"Bearer {o.json()['access_token']}"}
    assert (await client.get("/saved", headers=h2)).json() == []
    assert (await client.get("/chat/history", headers=h2)).json() == []


@pytest.mark.db
async def test_chat_rate_limit_and_error_envelope(client, auth, fake_llm, monkeypatch):
    from app import ratelimit

    monkeypatch.setattr(ratelimit, "_limiter", SlidingWindow(2, 60))
    fake_llm.script = [LLMResult(text="a"), LLMResult(text="b"), LLMResult(text="c")]
    codes = [(await client.post("/chat", headers=auth, json={"message": "hi"})).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = await client.post("/chat", headers=auth, json={"message": "hi"})
    assert r.headers["retry-after"] and r.json()["error"]["code"] == "rate_limited"
    assert (await client.post("/chat", headers=auth, json={"message": ""})).json()["error"]["code"] in (
        "validation_error", "rate_limited")
