"""SCRUM-493: request log rows are not lost on temporary failures or restarts, and never hold credentials."""

from __future__ import annotations

import threading
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from application.access_control_services.request_log import build_record, clean_query, is_secret_param
from infrastructure.external_integrations import supabase_request_log as store

CONFIG = ("https://db.example", "publishable")
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl"


class _Ok:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return HTTPError("https://db.example/rest/v1/request_log", code, "x", {}, None)


@pytest.mark.parametrize(
    "name,value",
    [("access_token", "a"), ("id_token", "a"), ("X-Api-Key", "a"), ("apiKey", "a"), ("client_secret", "a"),
     ("password", "a"), ("code", "oauth"), ("jwt", "a"), ("q", JWT)],
)
def test_credential_like_parameters_are_detected(name, value):
    assert is_secret_param(name, value)


def test_normal_filters_are_kept_and_credentials_dropped():
    assert not is_secret_param("season", "2024-25")
    query = f"season=2024-25&our=TOR&id_token=abc&next={JWT}&client_secret=s"
    assert clean_query(query) == "season=2024-25&our=TOR"


def test_record_never_contains_the_bearer_token():
    row = build_record(email="a@team.test", role="coach", method="get", path="/rank-plays/baseline",
                       raw_query=f"season=2024-25&access_token={JWT}", status_code=200, duration_ms=12.4).to_row()
    assert JWT not in str(row) and row["endpoint"] == "/rank-plays/baseline" and row["status_code"] == 200


def test_a_temporary_failure_is_retried_once():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError("reset")
        return _Ok()

    with patch.object(store, "_config", return_value=CONFIG), patch.object(store, "urlopen", fake_urlopen), \
            patch.object(store, "RETRY_DELAY_SECONDS", 0):
        assert store._insert("t", {"endpoint": "/x"}) is True
    assert len(calls) == 2


@pytest.mark.parametrize("code,expected_calls", [(503, 2), (429, 2), (401, 1), (403, 1)])
def test_only_server_errors_and_rate_limits_are_retried(code, expected_calls):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        raise http_error(code)

    with patch.object(store, "_config", return_value=CONFIG), patch.object(store, "urlopen", fake_urlopen), \
            patch.object(store, "RETRY_DELAY_SECONDS", 0):
        assert store._insert("t", {"endpoint": "/x"}) is False
    assert len(calls) == expected_calls


def test_flush_waits_for_queued_writes_before_shutdown():
    release = threading.Event()
    written = []

    def slow_insert(token, row):
        release.wait(2)
        written.append(row)
        return True

    with patch.object(store, "_config", return_value=CONFIG), patch.object(store, "_insert", slow_insert):
        store.submit_record("t", {"endpoint": "/a"})
        assert store.flush(timeout=0.05) is False  # still writing
        release.set()
        assert store.flush(timeout=2) is True
    assert written == [{"endpoint": "/a"}]


def test_shutdown_flushes_the_request_log():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from application.api_coordination.request_log_endpoints import install_request_logging

    app = FastAPI()
    install_request_logging(app)
    with patch.object(store, "flush") as flush:
        with TestClient(app):
            flush.assert_not_called()
    flush.assert_called_once()
