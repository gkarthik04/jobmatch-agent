"""Consistent error envelope: {"error": {"code", "message", "request_id", "details"?}}."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_conf import request_id_var

log = logging.getLogger("app.errors")


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
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, "http_error", str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return error_response(422, "validation_error", "Invalid request", details)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "Something went wrong")
