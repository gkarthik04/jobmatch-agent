# JobMatch Agent — architecture notes

AI career agent: resume upload → structured profile → hybrid-search job matches → tool-calling chat agent.
Mini version of Elip's product. Python 3.11, FastAPI (async), SQLAlchemy 2 async + Alembic, Postgres + pgvector (Neon), Gemini API.

## Layout
- `app/main.py` app factory, request-id + timeout middleware, error envelope, static SPA (`app/static/index.html`).
- `app/config.py` all settings from env (`.env`); model names / providers / RRF weights live here.
- `app/db.py` lazy async engine; `normalize_url` converts Neon URLs for asyncpg (ssl, no statement cache: PgBouncer).
- `app/models.py` ORM tables. `alembic/versions/` is the only way schema changes (0001 ext, 0002 tables, 0003 dim, 0004 title_tsv).
- `app/api/*` thin routers (auth, resume, matches/feedback/saved, search, chat, health). `app/services/*` all logic:
  - `llm.py` Gemini wrapper: ordered model fallback chain, per-minute limiter, JSON-schema + function calling, `cached_json`.
  - `embeddings.py` `LocalEmbedder` (fastembed bge-small, default) / `GeminiEmbedder` (1,000 texts/day free cap) + cache.
  - `search.py` vector + full-text + title-full-text → weighted RRF; `rerank.py` LLM rerank (fails open).
  - `resume.py` PDF → `ProfileSchema` with validate/retry; `matching.py` matches + explanations; `preferences.py` Rocchio.
  - `agent.py` explicit plan→act→check loop; `agent_tools.py` tool registry + Pydantic arg models.
  - `ingest.py` clean / upsert / embed only new-or-changed rows; `normalize_link` makes dataset links safe (http/https only).
  - `live_feed.py` Arbeitnow feed (open jobs + apply links, `ingest-live` CLI, daily action); `job_status.py` defines what "open" means (`is_open`, `OPEN_SQL`) and where the apply link points.
- `cli.py` fetch / ingest / stats. `evals/` offline suite (`python -m evals.run` → `evals/RESULTS.md`).
- `tests/` pytest; LLM and embedder are fakes. Tests marked `db` need `TEST_DATABASE_URL` (pgvector); CI provides it.

## Conventions
- Errors: raise `AppError(status, code, message)`; envelope is `{"error": {code, message, request_id}}`.
- Every Gemini call goes through `app/services/llm.py`; never call the SDK elsewhere.
- Changing `EMBED_PROVIDER/EMBED_MODEL/EMBED_DIM` requires `alembic upgrade head` (0003 resizes + drops old vectors) and re-running `cli.py ingest --add 0` to re-embed.
- Eval numbers are only ever written by `evals/run.py`; never hand-edit `evals/RESULTS.md`.
- No secrets in code. `.env` is git-ignored. `PREP.md` is git-ignored.

## Gotchas learned the hard way
- Free-tier quotas are per model and tiny; check `classify_error` before assuming a bug.
- Windows: write text files with `encoding="utf-8"`; `source .env` breaks on `&` in the Neon URL.
- asyncpg behind Neon's pooler needs `statement_cache_size=0`.

## Commands
- `ruff check .` · `pytest -q` · `alembic upgrade head` · `uvicorn app.main:app --reload`
