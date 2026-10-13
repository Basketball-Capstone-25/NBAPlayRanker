"""Data-quality reporting, source matching and analyst access."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.data_quality_endpoints import (
    EXPECTED_SEASONS,
    EXPECTED_TEAMS,
    create_data_quality_router,
)

AUTH_SERVICE = "application.access_control_services.access_control_service"
HEADERS = {"Authorization": "Bearer test"}


@pytest.fixture
def source():
    return pd.DataFrame([
        {
            "SEASON": season, "SEASON_ID": 22019 + index,
            "PLAYER_ID": 1, "TEAM_ABBREVIATION": team,
            "PLAY_TYPE": "Spotup", "TYPE_GROUPING": "Offensive",
            "POSS": 10, "PPP": 1.1, "PTS": 11,
        }
        for index, season in enumerate(EXPECTED_SEASONS)
        for team in EXPECTED_TEAMS
    ])


def _client(source):
    app = FastAPI()
    app.include_router(create_data_quality_router(source))
    return TestClient(app)


def _get(client, role="analyst"):
    with patch(f"{AUTH_SERVICE}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH_SERVICE}.get_profile_role", return_value=role,
    ):
        return client.get("/data/quality-report", headers=HEADERS)


def test_report_has_all_checks_source_metadata_and_no_cache(source):
    response = _get(_client(lambda: source))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    report = response.json()
    assert report["status"] == "pass"
    assert report["safe_for_use"] is True
    assert {check["name"]: check["status"] for check in report["checks"]} == {
        "schema": "pass", "completeness": "pass", "duplicates": "pass", "coverage": "pass",
    }
    assert report["dataset"]["rows"] == 180
    assert report["dataset"]["seasons"] == list(EXPECTED_SEASONS)
    assert report["dataset"]["teams"] == list(EXPECTED_TEAMS)
    assert report["dataset"]["version"].startswith("frame-sha256:")
    assert datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00")).tzinfo


def test_fingerprint_is_stable_and_changes_with_the_checked_source(source):
    client = _client(lambda: source)
    first = _get(client).json()
    assert _get(client).json()["dataset"]["version"] == first["dataset"]["version"]
    source.loc[0, "PPP"] = None
    changed = _get(client).json()
    assert changed["dataset"]["version"] != first["dataset"]["version"]
    assert changed["status"] == "fail"
    completeness = next(check for check in changed["checks"] if check["name"] == "completeness")
    assert completeness["details"]["null_counts"] == {"PPP": 1}


@pytest.mark.parametrize("missing", ["team", "season", "all_rows", "column"])
def test_missing_source_data_cannot_pass(source, missing):
    if missing == "team":
        source = source[source["TEAM_ABBREVIATION"] != "TOR"]
    elif missing == "season":
        source = source[source["SEASON"] != "2024-25"]
    elif missing == "all_rows":
        source = source.iloc[:0]
    else:
        source = source.drop(columns="PPP")
    response = _get(_client(lambda: source))
    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "fail"
    assert report["safe_for_use"] is False
    if missing == "column":
        assert next(check for check in report["checks"] if check["name"] == "completeness")["status"] == "not_run"


def test_duplicate_rows_are_reported(source):
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    report = _get(_client(lambda: source)).json()
    assert report["safe_for_use"] is False
    duplicates = next(check for check in report["checks"] if check["name"] == "duplicates")
    assert duplicates["status"] == "fail"
    assert duplicates["details"]["duplicate_rows"] == 1


@pytest.mark.parametrize("role", ["coach", None])
def test_only_analysts_can_read_report(source, role):
    client = _client(lambda: source)
    assert _get(client, role=role).status_code == 403
    assert client.get("/data/quality-report").status_code == 401


def test_internal_failure_does_not_expose_paths_or_details():
    def unavailable():
        raise FileNotFoundError("/private/source.csv secret-token")

    response = _get(_client(unavailable))
    assert response.status_code == 503
    assert response.json() == {"detail": "Data quality checks are unavailable. Please try again."}
    assert response.headers["Cache-Control"] == "no-store"


def test_app_route_reads_the_loaded_table_source(source, monkeypatch):
    monkeypatch.setattr(app_module.rec, "raw_df", source)
    response = _get(TestClient(app_module.app))
    assert response.status_code == 200
    assert response.json()["dataset"]["rows"] == len(source)
    assert response.json()["dataset"]["version"] == _get(_client(lambda: source)).json()["dataset"]["version"]
