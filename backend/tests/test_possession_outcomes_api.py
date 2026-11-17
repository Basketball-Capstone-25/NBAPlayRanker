from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.data_quality_endpoints import install_data_quality_header
from application.api_coordination.possession_outcomes_endpoints import create_possession_outcomes_router
from domain.baseline_recommendation.baseline_recommender import regular_season_rows

AUTH = "application.access_control_services.access_control_service"
PARAMS = {"season": "2024-25", "team": "TOR"}
PATH = "/analytics/playtypes/possession-outcomes"


def request(params=None, *, role="coach", token=True, mounted=False):
    app = app_module.app if mounted else FastAPI()
    if not mounted:
        app.include_router(create_possession_outcomes_router(app_module.rec, app_module.data_quality_gate))
        install_data_quality_header(app, app_module.data_quality_gate, [PATH])
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "possession-outcomes-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        return TestClient(app).get(PATH, params=PARAMS if params is None else params,
            headers={"Authorization": "Bearer test"} if token else {})


@pytest.mark.parametrize("mounted", [False, True])
def test_coach_receives_independent_rates_with_source_counts(mounted):
    response = request(mounted=mounted)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    body = response.json()
    assert body["filters"] == {**PARAMS, "side": "offense", "season_phase": "regular_season"}
    assert len(body["rows"]) == 11
    cut = next(row for row in body["rows"] if row["play_type"] == "Cut")
    assert cut["possessions"] == 745 and cut["rates"]["scoring"]["value"] == pytest.approx(0.6038657718120806)


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("analyst", True, 403),
    ("pending", True, 403), (None, True, 403)])
def test_outcomes_require_coach_access(role, token, status):
    assert request(role=role, token=token).status_code == status


@pytest.mark.parametrize("change,status", [({"team": "XXX"}, 400), ({"season": "2024-99"}, 400),
    ({"season": "2028-29"}, 400), ({"team": ""}, 422), ({"team": "XXXX"}, 422), ({"season": "bad"}, 422)])
def test_invalid_filters_are_rejected(change, status):
    assert request({**PARAMS, **change}).status_code == status


def test_missing_required_team_is_rejected():
    assert request({"season": "2024-25"}).status_code == 422


def test_missing_offensive_profile_is_explicit(monkeypatch):
    frame = app_module.rec.raw_df
    monkeypatch.setattr(app_module.rec, "raw_df", frame.loc[~(
        frame.TEAM_ABBREVIATION.eq("TOR") & frame.SEASON.eq("2024-25") & frame.TYPE_GROUPING.eq("Offensive"))].copy())
    response = request()
    assert response.status_code == 200
    assert response.json()["status"] == "no_data" and response.json()["rows"] == []


def test_rate_flags_do_not_weaken_existing_startup_validation(monkeypatch):
    frame = app_module.rec.raw_df.copy(deep=True)
    selected = frame.TEAM_ABBREVIATION.eq("TOR") & frame.SEASON.eq("2024-25") & frame.PLAY_TYPE.eq("Cut")
    frame.loc[selected, "SCORE_POSS_PCT"] = 1.2
    frame.loc[selected, "FT_POSS_PCT"] = float("nan")
    with pytest.raises(ValueError):
        regular_season_rows(frame)
    monkeypatch.setattr(app_module.rec, "raw_df", frame)
    response = request()
    assert response.status_code == 200
    cut = next(row for row in response.json()["rows"] if row["play_type"] == "Cut")
    assert cut["rates"]["scoring"]["status"] == "invalid" and cut["rates"]["scoring"]["value"] is None
    assert cut["rates"]["scoring"]["invalid_possessions"] == 745
    assert cut["rates"]["free_throw"]["status"] == "missing" and cut["rates"]["turnover"]["status"] == "ok"


def test_broken_counts_and_quality_failure_still_block(monkeypatch):
    frame = app_module.rec.raw_df.copy(deep=True)
    frame.loc[frame.TEAM_ABBREVIATION.eq("TOR"), "POSS"] = -1
    monkeypatch.setattr(app_module.rec, "raw_df", frame)
    bad_source = request()
    assert bad_source.status_code == 503
    assert bad_source.headers["Cache-Control"] == "private, no-store"
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {
        "status": "fail", "checks": [{"status": "fail", "name": "fixture"}]})
    response = request()
    assert response.status_code == 503 and response.headers["X-Data-Quality"] == "fail"
