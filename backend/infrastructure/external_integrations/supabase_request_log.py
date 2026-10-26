"""SCRUM-494: write and read public.request_log through Supabase's REST API.

Every call uses the caller's own access token, so row-level security decides
what is allowed: users add rows about themselves, analysts read them.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_WRITER = ThreadPoolExecutor(max_workers=2, thread_name_prefix="request-log")
TIMEOUT_SECONDS = 5


class RequestLogUnavailable(RuntimeError):
    """The log could not be read (not configured, refused or unreachable)."""


def _config() -> Optional[Tuple[str, str]]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "")
    return (url, key) if url and key else None


def _headers(token: str, key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "apikey": key, "Accept": "application/json"}


def _insert(token: str, row: Dict[str, Any]) -> None:
    config = _config()
    if config is None:
        return
    url, key = config
    request = Request(
        f"{url}/rest/v1/request_log",
        data=json.dumps(row).encode("utf-8"),
        headers={**_headers(token, key), "Content-Type": "application/json", "Prefer": "return=minimal"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS):
            pass
    except (HTTPError, URLError, OSError) as exc:
        # Logging must never break or slow a coach's request; record only the failure type.
        logger.warning("request log write failed (%s)", type(exc).__name__)


def submit_record(token: str, row: Dict[str, Any]) -> None:
    """Write in the background so the API response is not delayed."""
    if _config() is None:
        return
    _WRITER.submit(_insert, token, row)


def fetch_page(token: str, params: Sequence[tuple]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (rows, Content-Range header) for one page of the log."""
    config = _config()
    if config is None:
        raise RequestLogUnavailable("The request log is not configured on this server.")
    url, key = config
    request = Request(
        f"{url}/rest/v1/request_log?{urlencode(list(params))}",
        headers={**_headers(token, key), "Prefer": "count=exact"},
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            rows = json.load(response)
            content_range = response.headers.get("Content-Range")
    except HTTPError as exc:
        raise RequestLogUnavailable(f"The request log refused the query ({exc.code}).") from exc
    except (URLError, OSError, ValueError) as exc:
        raise RequestLogUnavailable("The request log could not be reached.") from exc
    if not isinstance(rows, list):
        raise RequestLogUnavailable("The request log returned an unexpected response.")
    return rows, content_range
