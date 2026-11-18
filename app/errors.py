"""Consistent error envelope: {"error": {"code", "message", "request_id", "details"?}}."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_conf import request_id_var

log = logging.getLogger("app.errors")
NOT_FOUND_PAGE = Path(__file__).parent / "static" / "404.html"


class AppError(Exception):
    def __init__(
        self, status: int, code: str, message: str, details: object = None,
        headers: dict[str, str] | None = None,
    ):
        self.status, self.code, self.message, self.details = status, code, message, details
        self.headers = headers
        super().__init__(message)


def error_response(status: int, code: str, message: str, details: object = None, headers=None):
    body: dict = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # browsers navigating to an unknown URL get the branded page; API clients keep JSON
        if exc.status_code == 404 and "text/html" in request.headers.get("accept", "") \
                and NOT_FOUND_PAGE.exists():
            return FileResponse(NOT_FOUND_PAGE, status_code=404)
        return error_response(exc.status_code, "http_error", str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        message = "Invalid request"
        if details:  # surface the first problem in plain words
            first = str(details[0]["msg"]).removeprefix("Value error, ")
            bad_email = first.startswith("value is not a valid email")
            message = "Enter a valid email address." if bad_email else first
        return error_response(422, "validation_error", message, details)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "Something went wrong")
