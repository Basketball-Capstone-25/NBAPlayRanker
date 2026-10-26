"""SCRUM-494 / UC17: endpoint activity log — recording rules, analyst search and access."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.access_control_services.request_log import (
    InvalidLogFilter,
    build_record,
    clean_query,
    page_payload,
    parse_filters,
    postgrest_params,
    should_log,
    total_from_content_range,
)
from application.api_coordination.app import app
from infrastructure.external_integrations import supabase_request_log as store

client = TestClient(app)
SERVICE = "application.access_control_services.access_control_service"
STORE = "application.api_coordination.request_log_endpoints.store"


def as_role(role):
    """Pretend a verified user with this database role is calling."""
    return patch(f"{SERVICE}.decode_supabase_jwt", return_value={"sub": "user-1", "email": "a@team.test", "exp": 9999999999}), patch(
        f"{SERVICE}.get_profile_role", return_value=role
    )


# Recording rules ---------------------------------------------------------

def test_only_signed_in_coach_and_analyst_calls_are_logged():
    assert should_log("GET", "/rank-plays/baseline", "coach")
    assert should_log("POST", "/nlp/parse", "analyst")
    assert not should_log("GET", "/rank-plays/baseline", None)
    assert not should_log("GET", "/health", "coach")
    assert not should_log("OPTIONS", "/rank-plays/baseline", "coach")


def test_credentials_are_removed_from_the_logged_query():
    assert clean_query("season=2024-25&token=abc&our=TOR&apikey=x") == "season=2024-25&our=TOR"
    assert len(clean_query("q=" + "x" * 900)) == 500


def test_record_has_who_what_when_fields():
    record = build_record(email="Coach@Team.test ", role="coach", method="get", path="/rank-plays/quick-call",
                          raw_query="season=2024-25", status_code=200, duration_ms=41.6)
    assert record.to_row() == {
        "user_email": "Coach@Team.test", "user_role": "coach", "method": "GET",
        "endpoint": "/rank-plays/quick-call", "query": "season=2024-25", "status_code": 200, "duration_ms": 42,
    }


# Filters and paging ------------------------------------------------------

def test_filters_become_newest_first_paged_postgrest_query():
    filters = parse_filters(user="coach@", endpoint="/rank-plays", date_from=date(2026, 10, 1),
                            date_to=date(2026, 10, 26), page=3, page_size=20)
    params = postgrest_params(filters)
    assert ("order", "created_at.desc,id.desc") in params
    assert ("limit", "20") in params and ("offset", "40") in params
    assert ("user_email", "ilike.*coach@*") in params
    assert ("endpoint", "ilike.*/rank-plays*") in params
    assert ("created_at", "gte.2026-10-01T00:00:00+00:00") in params
    assert ("created_at", "lt.2026-10-27T00:00:00+00:00") in params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": 0},
        {"page_size": 101},
        {"date_from": date(2026, 10, 2), "date_to": date(2026, 10, 1)},
        {"date_from": date(2026, 1, 1), "date_to": date(2026, 10, 1)},
        {"user": "a,b"},
        {"endpoint": "x)"},
    ],
)
def test_bad_filters_are_rejected(kwargs):
    with pytest.raises(InvalidLogFilter):
        parse_filters(**kwargs)


def test_page_reports_total_and_next_page():
    filters = parse_filters(page=2, page_size=2)
    assert total_from_content_range("2-3/5") == 5
    assert total_from_content_range("*/0") == 0
    assert page_payload([{"id": 3}, {"id": 4}], 5, filters)["has_next"] is True
    assert page_payload([{"id": 3}], 3, filters)["has_next"] is False


# API access --------------------------------------------------------------

def test_signed_out_user_cannot_view_the_log():
    assert client.get("/admin/request-log").status_code == 401


@pytest.mark.parametrize("role", ["coach", None])
def test_other_roles_get_403(role):
    decode, get_role = as_role(role)
    with decode, get_role, patch(f"{STORE}.fetch_page") as fetch:
        response = client.get("/admin/request-log", headers={"Authorization": "Bearer t"})
    assert response.status_code == 403
    fetch.assert_not_called()


def test_analyst_lists_filtered_page_with_their_own_token():
    rows = [{"id": 9, "created_at": "2026-10-26T15:00:00+00:00", "user_email": "c@team.test", "user_role": "coach",
             "method": "GET", "endpoint": "/rank-plays/quick-call", "query": "", "status_code": 200, "duration_ms": 40}]
    decode, get_role = as_role("analyst")
    with decode, get_role, patch(f"{STORE}.fetch_page", return_value=(rows, "0-0/1")) as fetch, patch(f"{STORE}.submit_record"):
        response = client.get(
            "/admin/request-log",
            params={"user": "c@", "endpoint": "/rank-plays", "date_from": "2026-10-20", "date_to": "2026-10-26", "page_size": 10},
            headers={"Authorization": "Bearer analyst-token"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == rows and body["total"] == 1 and body["has_next"] is False
    token, params = fetch.call_args.args
    assert token == "analyst-token"
    assert ("endpoint", "ilike.*/rank-plays*") in params and ("limit", "10") in params


def test_bad_filter_returns_400_and_store_outage_returns_503():
    decode, get_role = as_role("analyst")
    with decode, get_role, patch(f"{STORE}.submit_record"):
        bad = client.get("/admin/request-log", params={"date_from": "2026-10-26", "date_to": "2026-10-01"},
                         headers={"Authorization": "Bearer t"})
        with patch(f"{STORE}.fetch_page", side_effect=store.RequestLogUnavailable("down")):
            down = client.get("/admin/request-log", headers={"Authorization": "Bearer t"})
    assert bad.status_code == 400
    assert down.status_code == 503


# Recording middleware ----------------------------------------------------

def test_a_signed_in_call_queues_one_log_row_without_the_token():
    decode, get_role = as_role("analyst")
    with decode, get_role, patch(f"{STORE}.fetch_page", return_value=([], "*/0")), \
         patch("application.api_coordination.request_log_endpoints.decode_supabase_jwt", return_value={"email": "a@team.test"}), \
         patch(f"{STORE}.submit_record") as submit:
        client.get("/admin/request-log?token=leak", headers={"Authorization": "Bearer analyst-token"})
    submit.assert_called_once()
    token, row = submit.call_args.args
    assert token == "analyst-token"
    assert row["user_email"] == "a@team.test" and row["user_role"] == "analyst"
    assert row["endpoint"] == "/admin/request-log" and row["status_code"] == 200
    assert "leak" not in row["query"]


def test_unauthenticated_calls_are_not_logged():
    with patch(f"{STORE}.submit_record") as submit:
        client.get("/admin/request-log")
        client.get("/health")
    submit.assert_not_called()


def test_store_is_silent_when_not_configured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    store.submit_record("t", {"endpoint": "/x"})  # must not raise or block
    with pytest.raises(store.RequestLogUnavailable):
        store.fetch_page("t", [])
