import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import health
from app.config import get_settings
from app.db import dispose_engine
from app.errors import error_response, install_error_handlers
from app.logging_conf import request_id_var, setup_logging

log = logging.getLogger("app.request")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="JobMatch Agent", version="0.1.0", lifespan=lifespan)
    install_error_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(call_next(request), settings.request_timeout_s)
        except TimeoutError:
            response = error_response(504, "timeout", "Request timed out")
        finally:
            request_id_var.reset(token)
        ms = round((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = rid
        log.info(
            "request",
            extra={"method": request.method, "path": request.url.path,
                   "status": response.status_code, "ms": ms, "request_id": rid},
        )
        return response

    app.include_router(health.router)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
