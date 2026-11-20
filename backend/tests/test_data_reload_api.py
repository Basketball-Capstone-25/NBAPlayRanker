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


# SCRUM-491: the play-by-play reload uses the same role rules and error handling.
from application.data_reload_services.pbp_reload import PbpReloadService  # noqa: E402
from application.data_reload_services.synergy_reload import DataReloadError  # noqa: E402


@pytest.mark.parametrize("role", ["coach", None])
def test_only_analysts_can_reload_shots(role, monkeypatch):
    monkeypatch.setattr(PbpReloadService, "_reload", lambda self: pytest.fail("reload must not run"))
    assert call("POST", "/data/reload/pbp", role).status_code == 403


def test_analyst_shot_reload_returns_the_new_version(monkeypatch):
    result = {"dataset": "pbp_shots", "status": "reloaded", "previous_version": "a", "dataset_version": "b"}
    monkeypatch.setattr(PbpReloadService, "_reload", lambda self: result)
    response = call("POST", "/data/reload/pbp", "analyst")
    assert response.status_code == 200 and response.json() == result
    assert response.headers["Cache-Control"] == "no-store"


def test_failed_shot_reload_explains_and_keeps_previous_data(monkeypatch):
    def reject(self):
        raise DataReloadError("shots_clean.parquet is missing columns: GAME_ID.")

    monkeypatch.setattr(PbpReloadService, "_reload", reject)
    response = call("POST", "/data/reload/pbp", "analyst")
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Reload failed; the previous data is still in use. shots_clean.parquet is missing columns: GAME_ID."
    )


def test_shot_reload_rejects_self_opponent_with_422(tmp_path, monkeypatch):
    import pandas as pd
    from infrastructure.data_access import pbp_loader

    source = tmp_path / "shots_clean.parquet"
    pd.DataFrame([{
        "SEASON_STR": "2024-25", "TEAM_ABBR": "TOR", "OPP_ABBR": "TOR",
        "GAME_ID": "game-1", "X": 1.0, "Y": 4.0, "POINTS": 2,
    }]).to_parquet(source, index=False)
    service = PbpReloadService(clean_path=source, canonical_path=tmp_path / "canonical.parquet",
                               meta_path=tmp_path / "metadata.json")
    checked_reload = service._reload
    before = pbp_loader.loaded_pbp_canonical()
    monkeypatch.setattr(PbpReloadService, "_reload", lambda self: checked_reload())

    response = call("POST", "/data/reload/pbp", "analyst")

    assert response.status_code == 422
    assert "1 self-opponent rows" in response.json()["detail"]
    assert response.headers["Cache-Control"] == "no-store"
    assert pbp_loader.loaded_pbp_canonical() is before
    assert not service._canonical_path.exists() and not service._meta_path.exists()


@pytest.mark.parametrize("operation", ["write_json_atomic", "_read_parquet_batches"])
def test_shot_save_failure_returns_422_without_changing_data(tmp_path, monkeypatch, operation):
    import pandas as pd
    from infrastructure.data_access import pbp_loader, pbp_snapshot

    source = tmp_path / "shots_clean.parquet"
    pd.DataFrame([{
        "SEASON_STR": "2024-25", "TEAM_ABBR": "TOR", "OPP_ABBR": "BOS",
        "GAME_ID": "game-1", "X": 1.0, "Y": 4.0, "POINTS": 2,
    }]).to_parquet(source, index=False)
    service = PbpReloadService(clean_path=source, canonical_path=tmp_path / "canonical.parquet",
                               meta_path=tmp_path / "metadata.json")
    for path in (service._canonical_path, service._meta_path):
        path.write_bytes(b"previous snapshot")
    checked_reload = service._reload
    before = pbp_loader.loaded_pbp_canonical()
    monkeypatch.setattr(PbpReloadService, "_reload", lambda self: checked_reload())

    def fail(*args, **kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(pbp_snapshot, operation, fail)
    response = call("POST", "/data/reload/pbp", "analyst")

    assert response.status_code == 422
    assert "previous data is still in use" in response.json()["detail"]
    assert response.headers["Cache-Control"] == "no-store"
    assert pbp_loader.loaded_pbp_canonical() is before
    for path in (service._canonical_path, service._meta_path):
        assert path.read_bytes() == b"previous snapshot"
