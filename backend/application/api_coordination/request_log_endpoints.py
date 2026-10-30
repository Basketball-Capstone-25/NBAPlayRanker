"""SCRUM-494 / UC17: record endpoint activity and let analysts search it."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from application.access_control_services.request_log import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    InvalidLogFilter,
    build_record,
    page_payload,
    parse_filters,
    postgrest_params,
    should_log,
    total_from_content_range,
)
from application.api_coordination.auth_dependency import require_role
from infrastructure.external_integrations import supabase_request_log as store
from infrastructure.external_integrations.supabase_jwt import decode_supabase_jwt


def _flush_on_shutdown(app: FastAPI) -> None:
    """Wrap the app's lifespan so queued log writes finish before the process exits."""
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(running_app):
        async with previous(running_app) as state:
            yield state
        await run_in_threadpool(store.flush)

    app.router.lifespan_context = lifespan


def install_request_logging(app: FastAPI) -> None:
    """After each response, queue a log row for signed-in coach/analyst calls.

    SCRUM-493: queued rows are flushed on shutdown so a restart does not lose them.
    """
    _flush_on_shutdown(app)

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        audit = getattr(request.state, "audit_user", None)
        if audit and audit.get("token") and should_log(request.method, request.url.path, audit.get("role")):
            claims = decode_supabase_jwt(audit["token"]) or {}
            record = build_record(
                email=claims.get("email"),
                role=audit["role"],
                method=request.method,
                path=request.url.path,
                raw_query=request.url.query,
                status_code=response.status_code,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            store.submit_record(audit["token"], record.to_row())
        return response


def create_request_log_router() -> APIRouter:
    router = APIRouter(tags=["governance"])

    @router.get("/admin/request-log")
    def list_request_log(
        user: dict = Depends(require_role("analytics")),
        user_filter: str = Query("", alias="user", max_length=320, description="Part of a user's email."),
        endpoint: str = Query("", max_length=200, description="Part of an endpoint path, e.g. /rank-plays."),
        date_from: Optional[date] = Query(None, description="First day, YYYY-MM-DD (UTC)."),
        date_to: Optional[date] = Query(None, description="Last day, YYYY-MM-DD (UTC)."),
        page: int = Query(1, ge=1),
        page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    ) -> Dict[str, Any]:
        try:
            filters = parse_filters(
                user=user_filter, endpoint=endpoint, date_from=date_from, date_to=date_to,
                page=page, page_size=page_size,
            )
        except InvalidLogFilter as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        token = user.get("token")
        if not token:
            raise HTTPException(status_code=401, detail="Sign in to view the request log.")
        try:
            rows, content_range = store.fetch_page(token, postgrest_params(filters))
        except store.RequestLogUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return page_payload(rows, total_from_content_range(content_range), filters)

    return router
