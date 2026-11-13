from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module

AUTH = "application.access_control_services.access_control_service"
PARAMS = {"season": "2024-25", "team": "TOR", "side": "offense"}


def request(params=None, *, role="coach", token=True):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "league-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        return TestClient(app_module.app).get("/analytics/playtypes/league-comparison",
            params=PARAMS if params is None else params,
            headers={"Authorization": "Bearer test"} if token else {})


def test_coach_can_compare_one_team_without_an_opponent():
    response = request()
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    assert response.json()["filters"] == {**PARAMS, "season_phase": "regular_season"}


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("analyst", True, 403),
    ("pending", True, 403), (None, True, 403)])
def test_comparison_keeps_coach_access(role, token, status):
    assert request(role=role, token=token).status_code == status


@pytest.mark.parametrize("change,status", [({"season": "2024-99"}, 400), ({"team": "XXX"}, 400),
    ({"team": ""}, 422), ({"side": "both"}, 422), ({"season": "bad"}, 422)])
def test_comparison_rejects_invalid_filters(change, status):
    assert request({**PARAMS, **change}).status_code == status


def test_unknown_period_returns_no_data_instead_of_zero():
    response = request({**PARAMS, "season": "2028-29"})
    assert response.status_code == 200
    assert response.json()["status"] == "no_data"
    assert response.json()["rows"] == []


def test_bad_comparison_data_is_unavailable(monkeypatch):
    frame = app_module.rec.team_df.copy(deep=True)
    frame.loc[frame.TEAM_ABBREVIATION.eq("TOR"), "FG_PCT"] = float("nan")
    monkeypatch.setattr(app_module.rec, "team_df", frame)
    response = request()
    assert response.status_code == 503
    assert response.headers["Cache-Control"] == "private, no-store"


def test_comparison_uses_existing_quality_gate(monkeypatch):
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {
        "status": "fail", "checks": [{"status": "fail", "name": "fixture"}]})
    response = request()
    assert response.status_code == 503
    assert response.headers["X-Data-Quality"] == "fail"
