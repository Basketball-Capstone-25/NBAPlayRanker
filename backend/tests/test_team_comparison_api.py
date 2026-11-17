from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.data_quality_endpoints import install_data_quality_header
from application.api_coordination.team_comparison_endpoints import create_team_comparison_router

AUTH = "application.access_control_services.access_control_service"
PARAMS = {"season": "2024-25", "team": "TOR", "opponent": "BOS"}
PATH = "/analytics/playtypes/team-comparison"


def request(params=None, *, role="coach", token=True):
    app = FastAPI()
    app.include_router(create_team_comparison_router(app_module.rec, app_module.data_quality_gate))
    install_data_quality_header(app, app_module.data_quality_gate, [PATH])
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "team-comparison-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ):
        return TestClient(app).get(PATH, params=PARAMS if params is None else params,
            headers={"Authorization": "Bearer test"} if token else {})


def test_coach_receives_both_offensive_profiles():
    response = request()
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    body = response.json()
    assert body["filters"] == {**PARAMS, "side": "offense", "season_phase": "regular_season"}
    assert len(body["rows"]) == 11
    assert body["team_possessions"] > 0 and body["opponent_possessions"] > 0


def test_comparison_is_mounted_in_the_application():
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "team-comparison-test"}), patch(
        f"{AUTH}.get_profile_role", return_value="coach"
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        response = TestClient(app_module.app).get(PATH, params=PARAMS,
            headers={"Authorization": "Bearer test"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    assert response.json()["filters"] == {**PARAMS, "side": "offense", "season_phase": "regular_season"}
    assert len(response.json()["rows"]) == 11


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("analyst", True, 403),
    ("pending", True, 403), (None, True, 403)])
def test_profiles_require_coach_access(role, token, status):
    assert request(role=role, token=token).status_code == status


@pytest.mark.parametrize("change,status", [({"team": "XXX"}, 400), ({"opponent": "XXX"}, 400),
    ({"opponent": "TOR"}, 400), ({"season": "2024-99"}, 400), ({"season": "2028-29"}, 400),
    ({"team": ""}, 422), ({"opponent": "x" * 4}, 422), ({"season": "bad"}, 422)])
def test_invalid_profile_filters(change, status):
    assert request({**PARAMS, **change}).status_code == status


def test_missing_required_opponent_is_rejected():
    assert request({"season": "2024-25", "team": "TOR"}).status_code == 422


def test_missing_profile_preserves_the_other_team(monkeypatch):
    frame = app_module.rec.team_df
    monkeypatch.setattr(app_module.rec, "team_df", frame.loc[~(
        frame.TEAM_ABBREVIATION.eq("TOR") & frame.SEASON.eq("2024-25") & frame.SIDE.eq("offense"))].copy())
    response = request()
    assert response.status_code == 200 and response.json()["status"] == "no_data"
    assert response.json()["team_possessions"] is None
    assert all(row["team"] is None and row["opponent"] is not None for row in response.json()["rows"])


def test_bad_counts_and_quality_failure_block_the_comparison(monkeypatch):
    frame = app_module.rec.team_df.copy(deep=True)
    frame.loc[frame.TEAM_ABBREVIATION.eq("TOR"), "PTS"] = -1
    monkeypatch.setattr(app_module.rec, "team_df", frame)
    bad_source = request()
    assert bad_source.status_code == 503
    assert bad_source.headers["Cache-Control"] == "private, no-store"
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {
        "status": "fail", "checks": [{"status": "fail", "name": "fixture"}]})
    response = request()
    assert response.status_code == 503 and response.headers["X-Data-Quality"] == "fail"
