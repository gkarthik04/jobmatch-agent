# JobMatch Agent — architecture notes

AI career agent: resume upload → structured profile → hybrid-search job matches → tool-calling chat agent.
Mini version of Elip's product. Python 3.11, FastAPI (async), SQLAlchemy 2 async + Alembic, Postgres + pgvector (Neon), Gemini API.

## Layout
- `app/main.py` app factory, request-id + timeout middleware, error envelope, static SPA.
- `app/config.py` all settings from env (`.env`); model names live here, never hard-coded.
- `app/db.py` lazy async engine; `normalize_url` converts Neon URLs for asyncpg.
- `app/models.py` ORM tables. `alembic/versions/` is the only way schema changes.
- `app/api/*` thin routers. `app/services/*` all logic (llm, embeddings, search, rerank, resume, matching, preferences, agent).
- `cli.py` ingestion / embedding commands. `evals/` offline evaluation suite (`python -m evals.run`).
- `tests/` pytest; LLM is always mocked. Tests marked `db` need `TEST_DATABASE_URL`.

## Conventions
- Errors: raise `AppError(status, code, message)`; envelope is `{"error": {code, message, request_id}}`.
- Every Gemini call goes through `app/services/llm.py` (retry/backoff, cache, token logging).
- Embeddings are cached by content hash; only new/changed rows are re-embedded.
- No secrets in code. `.env` is git-ignored. `PREP.md` is git-ignored.

## Commands
- `ruff check .` · `pytest -q` · `alembic upgrade head` · `uvicorn app.main:app --reload`
