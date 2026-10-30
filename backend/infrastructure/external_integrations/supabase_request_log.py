"""SCRUM-494/493: write and read public.request_log through Supabase's REST API.

Every call uses the caller's own access token, so row-level security decides
what is allowed: users add rows about themselves, analysts read them.
Writes retry once on a temporary failure, and queued writes are flushed when
the backend shuts down so a restart does not drop them (SCRUM-493).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_WRITER = ThreadPoolExecutor(max_workers=2, thread_name_prefix="request-log")
TIMEOUT_SECONDS = 5
WRITE_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.5
_pending: "set[Future]" = set()
_pending_lock = threading.Lock()


class RequestLogUnavailable(RuntimeError):
    """The log could not be read (not configured, refused or unreachable)."""


def _config() -> Optional[Tuple[str, str]]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "")
    return (url, key) if url and key else None


def _headers(token: str, key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "apikey": key, "Accept": "application/json"}


def _retryable(exc: Exception) -> bool:
    """Network errors, rate limits and server errors may pass on a second try; RLS refusals will not."""
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (URLError, OSError))


def _insert(token: str, row: Dict[str, Any]) -> bool:
    config = _config()
    if config is None:
        return False
    url, key = config
    body = json.dumps(row).encode("utf-8")
    headers = {**_headers(token, key), "Content-Type": "application/json", "Prefer": "return=minimal"}
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        request = Request(f"{url}/rest/v1/request_log", data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS):
                return True
        except (HTTPError, URLError, OSError) as exc:
            if attempt < WRITE_ATTEMPTS and _retryable(exc):
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            # Logging must never break or slow a coach's request; record only the failure type.
            logger.warning("request log write failed (%s)", type(exc).__name__)
            return False
    return False


def submit_record(token: str, row: Dict[str, Any]) -> None:
    """Write in the background so the API response is not delayed."""
    if _config() is None:
        return
    future = _WRITER.submit(_insert, token, row)
    with _pending_lock:
        _pending.add(future)
    future.add_done_callback(_forget)


def _forget(future: Future) -> None:
    with _pending_lock:
        _pending.discard(future)


def flush(timeout: float = 10.0) -> bool:
    """Wait for queued log writes (called on shutdown). True when none are left."""
    with _pending_lock:
        waiting = list(_pending)
    if not waiting:
        return True
    _, not_done = wait(waiting, timeout=timeout)
    if not_done:
        logger.warning("request log flush timed out with %d writes pending", len(not_done))
    return not not_done


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
