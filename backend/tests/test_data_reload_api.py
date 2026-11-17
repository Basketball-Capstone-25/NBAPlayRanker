"""POST /data/reload/synergy: analyst only, no restart, old data kept on failure (SCRUM-490)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from domain.context_ml_recommendation import ml_context_recommender

AUTH_SERVICE = "application.access_control_services.access_control_service"
HEADERS = {"Authorization": "Bearer test"}
client = TestClient(app_module.app)


def call(method, path, role, **kwargs):
    with patch(f"{AUTH_SERVICE}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH_SERVICE}.get_profile_role", return_value=role,
    ):
        return client.request(method, path, headers=HEADERS, **kwargs)


@pytest.mark.parametrize("role", ["coach", None])
def test_only_analysts_can_reload(role):
    before = app_module.rec.raw_df
    response = call("POST", "/data/reload/synergy", role)
    assert response.status_code == 403
    assert app_module.rec.raw_df is before


def test_signed_out_reload_is_rejected():
    assert client.post("/data/reload/synergy").status_code == 401


def test_analyst_reload_updates_data_without_restart():
    before = app_module.rec.raw_df
    seasons, teams = app_module.VALID_SEASONS, app_module.VALID_TEAMS
    version = before.attrs["dataset_version"]

    response = call("POST", "/data/reload/synergy", "analyst")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reloaded" and body["dataset_version"] == version
    assert body["previous_version"] == version and body["quality_status"] == "pass"
    assert response.headers["Cache-Control"] == "no-store"
    # Same objects every router holds, now carrying the reloaded tables.
    assert app_module.rec.raw_df is not before
    assert app_module.VALID_SEASONS is seasons and seasons == body["seasons"]
    assert app_module.VALID_TEAMS is teams and "TOR" in teams
    assert app_module.ML_PRED_DF is not None
    assert ml_context_recommender._rec is app_module.rec
    # Coaches keep getting rankings from the reloaded tables.
    ranked = call("GET", "/rank-plays/baseline", "coach", params={"season": seasons[-1], "our": "TOR", "opp": "BOS", "k": 3})
    assert ranked.status_code == 200


def test_failed_reload_keeps_previous_data_and_explains(tmp_path, monkeypatch):
    before = app_module.rec.raw_df
    monkeypatch.setattr(app_module.rec, "source_path", tmp_path / "missing.csv")

    response = call("POST", "/data/reload/synergy", "analyst")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "previous data is still in use" in detail and "missing.csv was not found" in detail
    assert app_module.rec.raw_df is before
    ranked = call("GET", "/rank-plays/baseline", "coach", params={"season": app_module.VALID_SEASONS[-1], "our": "TOR", "opp": "BOS", "k": 3})
    assert ranked.status_code == 200
