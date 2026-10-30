"""SCRUM-627: Data Explorer play-by-play shots — filters, paging, export and access."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.analytics_services.shot_explorer import (
    EXPORT_LIMIT,
    InvalidShotFilter,
    ShotExplorerService,
    ShotFilters,
)
from application.api_coordination.shot_explorer_endpoints import create_shot_explorer_router

SERVICE = "application.access_control_services.access_control_service"


def shots(n_tor_bos: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_tor_bos):
        rows.append(dict(SEASON_STR="2024-25", TEAM_ABBR="TOR", OPP_ABBR="BOS", GAME_ID=401, PERIOD=4,
                         CLOCK_SEC=125 - i, SHOT_TYPE="Jump Shot", ZONE="Corner 3", IS_MAKE=i % 2 == 0,
                         POINTS=3 if i % 2 == 0 else 0, X=22.04, Y=1.0, DIST=22.0))
    rows.append(dict(SEASON_STR="2024-25", TEAM_ABBR="TOR", OPP_ABBR="NYK", GAME_ID=300, PERIOD=1,
                     CLOCK_SEC=700, SHOT_TYPE="Layup", ZONE="Restricted Area", IS_MAKE=True, POINTS=2,
                     X=float("nan"), Y=float("nan"), DIST=1.0))
    rows.append(dict(SEASON_STR="2023-24", TEAM_ABBR="TOR", OPP_ABBR="BOS", GAME_ID=200, PERIOD=2,
                     CLOCK_SEC=10, SHOT_TYPE="Layup", ZONE="Restricted Area", IS_MAKE=False, POINTS=0,
                     X=0.0, Y=2.0, DIST=2.0))
    df = pd.DataFrame(rows)
    df.attrs["dataset_version"] = "sha256:test"
    return df


def service(df=None):
    return ShotExplorerService(lambda: shots() if df is None else df)


# Filter rules -------------------------------------------------------------

def test_season_and_team_are_required():
    with pytest.raises(InvalidShotFilter, match="season and team"):
        ShotFilters.parse("", "TOR")
    with pytest.raises(InvalidShotFilter):
        ShotFilters.parse("2024-25", " ")


@pytest.mark.parametrize("kwargs", [dict(result="blocked"), dict(page=0), dict(page_size=0), dict(page_size=201)])
def test_bad_filters_are_rejected(kwargs):
    with pytest.raises(InvalidShotFilter):
        ShotFilters.parse("2024-25", "TOR", **kwargs)


def test_filters_are_normalized():
    f = ShotFilters.parse(" 2024-25 ", "tor", opp="bos", result="MADE", shot_type=" ")
    assert (f.team, f.opp, f.result, f.shot_type) == ("TOR", "BOS", "made", None)


def test_filters_work_together_and_count_matches():
    page = service().page(ShotFilters.parse("2024-25", "TOR", opp="BOS", result="made"))
    assert page["total"] == 2
    assert all(r["opponent"] == "BOS" and r["result"] == "Made" for r in page["rows"])


def test_rows_use_plain_language_columns_and_stable_order():
    page = service().page(ShotFilters.parse("2024-25", "TOR"))
    assert [c["label"] for c in page["columns"]][:3] == ["Game", "Quarter", "Time left"]
    assert [r["game_id"] for r in page["rows"]] == [300, 401, 401, 401]
    assert [r["time_left"] for r in page["rows"]][1:] == ["2:05", "2:04", "2:03"]
    first = page["rows"][0]
    assert first["x"] is None and first["y"] is None  # no valid court location, still counted
    assert page["rows"][1]["x"] == 22.0 and page["dataset_version"] == "sha256:test"


def test_paging_returns_page_size_rows_and_next_flag():
    svc = service(shots(120))
    first = svc.page(ShotFilters.parse("2024-25", "TOR", page=1))
    third = svc.page(ShotFilters.parse("2024-25", "TOR", page=3))
    assert (len(first["rows"]), first["total"], first["has_next"]) == (50, 121, True)
    assert (len(third["rows"]), third["has_next"]) == (21, False)


def test_no_matches_returns_empty_page():
    page = service().page(ShotFilters.parse("2024-25", "LAL"))
    assert page["total"] == 0 and page["rows"] == [] and page["has_next"] is False


def test_export_has_same_columns_as_table_and_reports_cap():
    big = shots(EXPORT_LIMIT + 5)
    export = service(big).export(ShotFilters.parse("2024-25", "TOR", opp="BOS"))
    assert export.capped and len(export.rows) == EXPORT_LIMIT and export.total == EXPORT_LIMIT + 5
    lines = export.to_csv().splitlines()
    assert "capped" in lines[0] and "team=TOR" in lines[0] and "sha256:test" in lines[0]
    assert lines[1].split(",") == [c["label"] for c in export.columns]
    body = export.to_json()
    assert body["capped"] and body["columns"] == export.columns and body["filters"]["opp"] == "BOS"


# API ---------------------------------------------------------------------

def app_with(svc) -> TestClient:
    app = FastAPI()
    app.include_router(create_shot_explorer_router(svc))
    return TestClient(app)


def as_role(role):
    return patch(f"{SERVICE}.decode_supabase_jwt", return_value={"sub": "u1", "exp": 9999999999}), patch(
        f"{SERVICE}.get_profile_role", return_value=role
    )


AUTH = {"Authorization": "Bearer t"}


def test_analyst_gets_a_page():
    a, b = as_role("analyst")
    with a, b:
        r = app_with(service()).get("/data/pbp-shots", params={"season": "2024-25", "team": "TOR"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["total"] == 4


@pytest.mark.parametrize("role", ["coach", "pending"])
def test_other_roles_get_403(role):
    a, b = as_role(role)
    with a, b:
        r = app_with(service()).get("/data/pbp-shots", params={"season": "2024-25", "team": "TOR"}, headers=AUTH)
    assert r.status_code == 403


def test_signed_out_gets_401():
    r = app_with(service()).get("/data/pbp-shots", params={"season": "2024-25", "team": "TOR"})
    assert r.status_code == 401


def test_bad_filter_is_400_and_missing_data_is_503():
    def missing():
        raise FileNotFoundError("shots_canonical.parquet not found")

    a, b = as_role("analyst")
    with a, b:
        bad = app_with(service()).get("/data/pbp-shots", params={"season": "2024-25", "team": "TOR", "result": "x"}, headers=AUTH)
        down = app_with(ShotExplorerService(missing)).get("/data/pbp-shots", params={"season": "2024-25", "team": "TOR"}, headers=AUTH)
    assert bad.status_code == 400 and "made or missed" in bad.json()["detail"]
    assert down.status_code == 503 and down.json()["detail"] == "Shot data is not available yet."


def test_csv_and_json_exports_download_with_filenames():
    a, b = as_role("analyst")
    params = {"season": "2024-25", "team": "TOR", "opp": "BOS"}
    with a, b:
        client = app_with(service())
        csv_r = client.get("/data/pbp-shots.csv", params=params, headers=AUTH)
        json_r = client.get("/data/pbp-shots.json", params=params, headers=AUTH)
    assert csv_r.headers["content-type"].startswith("text/csv")
    assert 'filename="pbp_shots_2024-25_TOR_BOS.csv"' in csv_r.headers["content-disposition"]
    assert json_r.headers["content-type"].startswith("application/json")
    assert json.loads(json_r.text)["exported_rows"] == 3
