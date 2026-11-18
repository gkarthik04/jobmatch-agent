import hashlib
import io
import json
import os
import re

import pytest

# Make sure tests never read a developer's real secrets/db by accident.
os.environ.setdefault("JWT_SECRET", "test-secret-key-that-is-at-least-32-bytes-long")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


# ---- fakes ------------------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing tokens get similar vectors."""

    def __init__(self, dim: int | None = None):
        from app.config import get_settings

        self.dim = dim or get_settings().embed_dim

    async def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        import math

        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM:
    """Answers by schema: profile / rerank / explain. For agent turns pops `script` results."""

    def __init__(self):
        self.script: list = []
        self.calls = 0

    async def generate(self, contents, *, system=None, schema=None, tools=None, temperature=0.0):
        from app.schemas import ProfileSchema
        from app.services.llm import LLMResult
        from app.services.matching import ExplainOut
        from app.services.rerank import RerankOut

        self.calls += 1
        if schema is ProfileSchema:
            return LLMResult(text=json.dumps({
                "name": "Test Candidate", "skills": ["Python", "SQL", "Docker", "python"],
                "experience_years": 1.5,
                "education": [{"degree": "B.Tech CSE", "institution": "XYZ", "year": 2025}],
                "projects": [{"name": "RAG bot", "description": "retrieval", "technologies": ["FAISS"]}],
                "preferred_roles": ["zyxwvu engineer"], "preferred_locations": ["Remote"],
            }))
        text = contents if isinstance(contents, str) else ""
        if schema is RerankOut:
            ids = [int(i) for i in re.findall(r"^\[(\d+)\]", text, re.M)]
            return LLMResult(text=json.dumps(
                {"scores": [{"id": i, "score": float(len(ids) - n)} for n, i in enumerate(ids)]}))
        if schema is ExplainOut:
            ids = [int(i) for i in re.findall(r"job_id=(\d+)", text)]
            return LLMResult(text=json.dumps({"items": [
                {"job_id": i, "why_it_matches": "Python and SQL overlap.", "fit_score": 80} for i in ids]}))
        return self.script.pop(0)


def make_pdf(lines: list[str]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for ln in lines:
        c.drawString(40, y, ln)
        y -= 16
    c.save()
    return buf.getvalue()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture(scope="session")
async def test_engine():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db import normalize_url

    url, connect_args = normalize_url(os.environ["TEST_DATABASE_URL"])
    engine = create_async_engine(url, connect_args=connect_args, poolclass=NullPool)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(test_engine, fake_llm, monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.db import get_session
    from app.deps import get_embedder, get_llm_dep
    from app.main import app
    from app.ratelimit import get_limiter

    maker = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _session():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: fake_llm
    get_limiter().reset()

    async def no_cache(session, key, produce):  # never write fake-LLM output to the shared cache
        return await produce()

    monkeypatch.setattr("app.services.rerank.cached_json", no_cache)
    monkeypatch.setattr("app.services.matching.cached_json", no_cache)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        c.maker = maker
        yield c
    app.dependency_overrides.clear()
    async with maker() as s:  # remove only what tests created
        await s.execute(text("DELETE FROM users WHERE email LIKE 'test-%@example.com'"))
        await s.execute(text("DELETE FROM jobs WHERE source_id LIKE 'test-%'"))
        await s.commit()


@pytest.fixture
async def seed_jobs(client):
    """Three searchable test jobs (unique token `zyxwvu`) so real data can't interfere."""
    from app.models import Job
    from app.services.ingest import embedding_text

    rows = [
        dict(source_id="test-1", title="Zyxwvu Backend Engineer", company="Acme", location="Austin, TX",
             remote=True, skills=["python", "sql", "docker"],
             description="zyxwvu backend python sql docker services " * 10),
        dict(source_id="test-2", title="Zyxwvu Data Analyst", company="Beta", location="Boston, MA",
             remote=False, skills=["sql", "tableau"],
             description="zyxwvu analytics sql tableau dashboards " * 10),
        dict(source_id="test-3", title="Zyxwvu Frontend Developer", company="Gamma", location="Austin, TX",
             remote=False, skills=["react", "javascript"],
             description="zyxwvu frontend react javascript ui " * 10),
    ]
    emb = FakeEmbedder()
    async with client.maker() as s:
        for r in rows:
            job = Job(content_hash=r["source_id"], **r)
            job.embedding = (await emb.embed([embedding_text(job)]))[0]
            s.add(job)
        await s.commit()
        from sqlalchemy import select

        ids = (await s.execute(select(Job.id).where(Job.source_id.like("test-%")).order_by(Job.id))).scalars().all()
    return ids


@pytest.fixture
async def auth(client):
    """Signed-up user with Authorization header."""
    import uuid

    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    r = await client.post("/auth/signup", json={"email": email, "password": "correct horse"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
