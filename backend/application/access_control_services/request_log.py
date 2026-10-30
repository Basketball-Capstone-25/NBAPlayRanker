"""SCRUM-494 / UC17: rules for recording and searching endpoint activity.

Pure functions only: what gets recorded, how filters are validated, and how a
page of results is described. HTTP calls live in
infrastructure/external_integrations/supabase_request_log.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import parse_qsl, urlencode

LOGGED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
UNLOGGED_PATHS = frozenset({"/health", "/ready", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/nlp/health"})
SECRET_PARAM_NAMES = frozenset({"token", "access_token", "refresh_token", "apikey", "api_key", "key", "password", "secret", "code", "sig", "signature"})
# SCRUM-493: any parameter whose name suggests a credential, or whose value looks like a JWT, is never stored.
SECRET_NAME_PARTS = ("token", "secret", "password", "passwd", "apikey", "api_key", "authorization", "jwt", "credential")
_JWT_VALUE = re.compile(r"^eyJ[\w-]+\.[\w-]+\.[\w-]*$")
QUERY_MAX_CHARS = 500
ENDPOINT_MAX_CHARS = 200
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_RANGE_DAYS = 92


@dataclass(frozen=True)
class RequestRecord:
    """One row of the request log, before the database adds id, user_id and time."""

    user_email: Optional[str]
    user_role: str
    method: str
    endpoint: str
    query: str
    status_code: int
    duration_ms: int

    def to_row(self) -> Dict[str, Any]:
        return {
            "user_email": self.user_email,
            "user_role": self.user_role,
            "method": self.method,
            "endpoint": self.endpoint,
            "query": self.query,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
        }


def should_log(method: str, path: str, role: Optional[str]) -> bool:
    """Only signed-in coach/analyst calls to real endpoints are recorded."""
    return role in ("coach", "analyst") and method.upper() in LOGGED_METHODS and path not in UNLOGGED_PATHS


def is_secret_param(name: str, value: str) -> bool:
    """True for credential-like parameters: known names, credential-like names or JWT values."""
    lowered = name.lower().replace("-", "_")
    if lowered in SECRET_PARAM_NAMES or any(part in lowered for part in SECRET_NAME_PARTS):
        return True
    return bool(_JWT_VALUE.match(value.strip()))


def clean_query(raw_query: str) -> str:
    """Drop credential-like parameters and cap the length."""
    if not raw_query:
        return ""
    kept = [(k, v) for k, v in parse_qsl(raw_query, keep_blank_values=True) if not is_secret_param(k, v)]
    text = urlencode(kept)
    return text[:QUERY_MAX_CHARS]


def build_record(
    *, email: Optional[str], role: str, method: str, path: str, raw_query: str, status_code: int, duration_ms: float
) -> RequestRecord:
    email = (email or "").strip()[:320] or None
    return RequestRecord(
        user_email=email,
        user_role=role,
        method=method.upper(),
        endpoint=(path or "/")[:ENDPOINT_MAX_CHARS],
        query=clean_query(raw_query),
        status_code=int(status_code),
        duration_ms=max(0, int(round(duration_ms))),
    )


class InvalidLogFilter(ValueError):
    """A search filter cannot be applied."""


@dataclass(frozen=True)
class LogFilters:
    user: str = ""
    endpoint: str = ""
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def _safe_text(value: str, label: str, limit: int) -> str:
    value = (value or "").strip()
    if len(value) > limit:
        raise InvalidLogFilter(f"{label} filter is too long.")
    # PostgREST filter syntax uses these characters; reject them instead of escaping.
    if any(ch in value for ch in ",()*\\\"") :
        raise InvalidLogFilter(f"{label} filter contains unsupported characters.")
    return value


def parse_filters(
    *, user: str = "", endpoint: str = "", date_from: Optional[date] = None, date_to: Optional[date] = None,
    page: int = 1, page_size: int = DEFAULT_PAGE_SIZE,
) -> LogFilters:
    if page < 1:
        raise InvalidLogFilter("Page must be 1 or more.")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise InvalidLogFilter(f"Page size must be between 1 and {MAX_PAGE_SIZE}.")
    if date_from and date_to:
        if date_from > date_to:
            raise InvalidLogFilter("The start date must be on or before the end date.")
        if (date_to - date_from).days > MAX_RANGE_DAYS:
            raise InvalidLogFilter(f"Choose a date range of {MAX_RANGE_DAYS} days or less.")
    return LogFilters(
        user=_safe_text(user, "User", 320),
        endpoint=_safe_text(endpoint, "Endpoint", ENDPOINT_MAX_CHARS),
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )


def postgrest_params(filters: LogFilters) -> List[tuple]:
    """Query parameters for GET /rest/v1/request_log (newest first)."""
    params: List[tuple] = [
        ("select", "id,created_at,user_email,user_role,method,endpoint,query,status_code,duration_ms"),
        ("order", "created_at.desc,id.desc"),
        ("limit", str(filters.page_size)),
        ("offset", str(filters.offset)),
    ]
    if filters.user:
        params.append(("user_email", f"ilike.*{filters.user}*"))
    if filters.endpoint:
        params.append(("endpoint", f"ilike.*{filters.endpoint}*"))
    if filters.date_from:
        start = datetime.combine(filters.date_from, datetime.min.time(), tzinfo=timezone.utc)
        params.append(("created_at", f"gte.{start.isoformat()}"))
    if filters.date_to:
        end = datetime.combine(filters.date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        params.append(("created_at", f"lt.{end.isoformat()}"))
    return params


def total_from_content_range(header: Optional[str]) -> Optional[int]:
    """'0-49/1234' -> 1234; '*/0' -> 0; unknown -> None."""
    if not header or "/" not in header:
        return None
    total = header.rsplit("/", 1)[1]
    return int(total) if total.isdigit() else None


def page_payload(rows: List[Mapping[str, Any]], total: Optional[int], filters: LogFilters) -> Dict[str, Any]:
    has_next = (filters.offset + len(rows) < total) if total is not None else len(rows) == filters.page_size
    return {
        "items": list(rows),
        "page": filters.page,
        "page_size": filters.page_size,
        "total": total,
        "has_next": has_next,
        "filters": {
            "user": filters.user,
            "endpoint": filters.endpoint,
            "date_from": filters.date_from.isoformat() if filters.date_from else None,
            "date_to": filters.date_to.isoformat() if filters.date_to else None,
        },
    }
