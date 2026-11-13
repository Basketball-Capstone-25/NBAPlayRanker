from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module

AUTH = "application.access_control_services.access_control_service"
PARAMS = {"team": "TOR", "play_type": "Cut"}


def request(params=None, *, role="coach", token=True):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "trend-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        return TestClient(app_module.app).get("/analytics/playtypes/season-trend",
            params=PARAMS if params is None else params,
            headers={"Authorization": "Bearer test"} if token else {})


def test_coach_gets_all_regular_seasons_without_an_opponent():
    response = request()
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    assert response.json()["filters"] == {**PARAMS, "side": "offense", "season_phase": "regular_season"}
    assert len(response.json()["rows"]) == 6


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("analyst", True, 403),
    ("pending", True, 403), (None, True, 403)])
def test_trend_requires_coach_access(role, token, status):
    assert request(role=role, token=token).status_code == status


@pytest.mark.parametrize("change,status", [({"team": "XXX"}, 400), ({"play_type": "Unknown"}, 400),
    ({"team": ""}, 422), ({"play_type": ""}, 422), ({"play_type": "x" * 41}, 422)])
def test_invalid_trend_filters(change, status):
    assert request({**PARAMS, **change}).status_code == status


def test_missing_category_returns_gaps(monkeypatch):
    frame = app_module.rec.team_df
    monkeypatch.setattr(app_module.rec, "team_df", frame.loc[
        ~(frame.TEAM_ABBREVIATION.eq("TOR") & frame.PLAY_TYPE.eq("Cut"))].copy())
    response = request()
    assert response.status_code == 200 and response.json()["status"] == "no_data"
    assert all(r["possessions"] is None for r in response.json()["rows"])


def test_invalid_counts_and_quality_failure_are_unavailable(monkeypatch):
    frame = app_module.rec.team_df.copy(deep=True)
    frame.loc[frame.TEAM_ABBREVIATION.eq("TOR"), "POSS"] = -1
    monkeypatch.setattr(app_module.rec, "team_df", frame)
    assert request().status_code == 503
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {
        "status": "fail", "checks": [{"status": "fail", "name": "fixture"}]})
    response = request()
    assert response.status_code == 503 and response.headers["X-Data-Quality"] == "fail"
