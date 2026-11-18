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
    assert args["ssl"] == "require" and args["statement_cache_size"] == 0


async def test_landing_page_and_legal_pages_are_served():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        home = await c.get("/")
        assert home.status_code == 200 and "text/html" in home.headers["content-type"]
        assert "Know which jobs" in home.text and "Upload your resume" in home.text
        assert 'application/ld+json' in home.text  # FAQ schema present
        for path in ("/privacy", "/terms", "/static/styles.css", "/static/app.js", "/static/og.png"):
            assert (await c.get(path)).status_code == 200, path
        assert "image/svg+xml" in (await c.get("/favicon.ico")).headers["content-type"]


async def test_unknown_page_gets_branded_404_for_browsers_but_json_for_api_clients():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        page = await c.get("/nope", headers={"accept": "text/html,application/xhtml+xml"})
        assert page.status_code == 404 and "That page does not exist" in page.text
        api = await c.get("/nope", headers={"accept": "application/json"})
        assert api.status_code == 404 and api.json()["error"]["code"] == "http_error"
