"""Rankings are blocked or flagged when the source fails its quality checks (SCRUM-479)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.data_quality_endpoints import DataQualityGate

AUTH_SERVICE = "application.access_control_services.access_control_service"
HEADERS = {"Authorization": "Bearer test"}
SEASON = app_module.VALID_SEASONS[-1]
MATCHUP = {"season": SEASON, "our": "TOR", "opp": "BOS", "k": 3}


def _get(path, params):
    client = TestClient(app_module.app)
    with patch(f"{AUTH_SERVICE}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH_SERVICE}.get_profile_role", return_value="coach",
    ):
        return client.get(path, params=params, headers=HEADERS)


@pytest.fixture
def failing_source(monkeypatch):
    """Same loaded data with one team removed, so the coverage check fails."""
    real = app_module.rec.raw_df
    broken = real[real["TEAM_ABBREVIATION"] != "TOR"]
    monkeypatch.setattr(app_module.rec, "raw_df", broken)
    yield broken
    app_module.data_quality_gate.refresh()


def test_checks_run_once_per_loaded_dataset():
    source = app_module.rec.raw_df
    gate = DataQualityGate(lambda: source)

    for _ in range(5):
        gate()

    assert gate.runs == 1
    assert gate.status() == "pass"


def test_gate_blocks_failed_source_and_names_the_failed_check(failing_source):
    gate = DataQualityGate(lambda: failing_source)

    with pytest.raises(HTTPException) as raised:
        gate()

    assert raised.value.status_code == 503
    assert "coverage" in raised.value.detail
    assert raised.value.headers["X-Data-Quality"] == "fail"


def test_rankings_carry_the_quality_header_when_data_passes():
    for path in ("/rank-plays/baseline", "/rank-plays/baseline.csv"):
        res = _get(path, MATCHUP)
        assert res.status_code == 200, res.text
        assert res.headers["X-Data-Quality"] == "pass"


@pytest.mark.parametrize("path,params", [
    ("/rank-plays/baseline", MATCHUP),
    ("/rank-plays/baseline.csv", MATCHUP),
    ("/rank-plays/context-ml", {**MATCHUP, "margin": -3, "period": 4, "time_remaining": 60}),
])
def test_rankings_are_blocked_when_source_fails(failing_source, path, params):
    res = _get(path, params)

    assert res.status_code == 503
    assert res.headers["X-Data-Quality"] == "fail"
    assert "coverage" in res.json()["detail"]


def test_flag_mode_serves_rankings_but_marks_them(failing_source, monkeypatch):
    monkeypatch.setenv("DATA_QUALITY_ENFORCEMENT", "flag")

    res = _get("/rank-plays/baseline", MATCHUP)

    assert res.status_code == 200, res.text
    assert res.headers["X-Data-Quality"] == "fail"
