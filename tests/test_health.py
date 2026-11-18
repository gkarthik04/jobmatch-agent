from httpx import ASGITransport, AsyncClient

from app.db import normalize_url
from app.main import app


async def test_health_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert r.headers["x-request-id"]


async def test_error_envelope_for_unknown_route():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/nope")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "http_error" and body["request_id"]


def test_normalize_neon_url():
    url, args = normalize_url(
        "postgresql://u:p@ep-x.neon.tech/db?sslmode=require&channel_binding=require"
    )
    assert url.startswith("postgresql+asyncpg://u:p@ep-x.neon.tech/db")
    assert "sslmode" not in url and "channel_binding" not in url
    assert args == {"ssl": "require"}
