"""SCRUM-629: late-game (clutch) report — filters, zone splits, small samples, exports and access."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from application.analytics_services.clutch_report import (
    SMALL_SAMPLE,
    ClutchDataUnavailable,
    ClutchFilters,
    InvalidClutchFilter,
    build_report,
    report_csv,
    report_filename,
    situation_text,
)

SERVICE = "application.access_control_services.access_control_service"


def shot(team, opp, zone, made, period=4, clock=120, margin=-2, points=None, season="2024-25"):
    value = 3 if zone in ("corner3", "arc3") else 2
    return dict(SEASON_STR=season, TEAM_ABBR=team, OPP_ABBR=opp, PERIOD=period, CLOCK_SEC=clock,
                MARGIN=margin, ZONE=zone, IS_MAKE=made, POINTS=(value if made else 0) if points is None else points)


def shots() -> pd.DataFrame:
    rows = [shot("TOR", "BOS", "corner3", i % 2 == 0) for i in range(SMALL_SAMPLE)]  # 20 shots, 10 makes
    rows += [shot("TOR", "BOS", "rim", True), shot("TOR", "BOS", "rim", False)]
    rows += [shot("BOS", "TOR", "rim", True, period=5, clock=30, margin=1)]  # overtime counts
    rows += [
        shot("TOR", "BOS", "mid", True, period=3),                 # too early
        shot("TOR", "BOS", "mid", True, clock=400),                # too much time left
        shot("TOR", "BOS", "mid", True, margin=12),                # game not close
        shot("TOR", "BOS", "mid", True, margin=float("nan")),      # unknown margin is left out
        shot("TOR", "NYK", "mid", True),                           # other opponent
        shot("TOR", "BOS", "mid", True, season="2023-24"),         # other season
    ]
    df = pd.DataFrame(rows)
    df.attrs["dataset_version"] = "sha256:test"
    return df


DEFAULT = ClutchFilters.parse("2024-25", "tor", "bos")


# Filters -----------------------------------------------------------------

def test_defaults_are_last_five_minutes_of_close_games():
    assert DEFAULT.as_dict() == {"season": "2024-25", "team": "TOR", "opp": "BOS", "period_from": 4,
                                 "seconds_left": 300, "margin_min": -5, "margin_max": 5}


@pytest.mark.parametrize("args, message", [
    (("", "TOR", "BOS"), "Pick a season."),
    (("2024-25", "TOR", ""), "Pick a opponent."),
    (("2024-25", "TOR", "tor"), "Pick two different teams."),
    (("2024-25", "TOR", "BOS", 0), "Start period"),
    (("2024-25", "TOR", "BOS", 4, 721), "Time left"),
    (("2024-25", "TOR", "BOS", 4, 300, -41, 5), "Score margin"),
    (("2024-25", "TOR", "BOS", 4, 300, 3, -3), "lowest margin"),
])
def test_bad_filters_are_rejected(args, message):
    with pytest.raises(InvalidClutchFilter, match=message):
        ClutchFilters.parse(*args)


# Report ------------------------------------------------------------------

def test_only_matching_late_game_shots_are_counted_for_each_team():
    report = build_report(shots(), DEFAULT)
    assert report["totals"]["team"]["shots"] == 22 and report["totals"]["opp"]["shots"] == 1
    assert [r["zone"] for r in report["rows"]] == ["rim", "corner3"]
    assert report["dataset_version"] == "sha256:test" and report["message"] is None


def test_zones_show_shots_makes_and_points_per_shot_side_by_side():
    rows = {r["zone"]: r for r in build_report(shots(), DEFAULT)["rows"]}
    assert rows["corner3"]["label"] == "Corner 3"
    assert rows["corner3"]["team"] == {"shots": 20, "makes": 10, "points": 30, "points_per_shot": 1.5, "small_sample": False}
    assert rows["corner3"]["opp"]["shots"] == 0 and rows["corner3"]["opp"]["points_per_shot"] is None
    assert rows["rim"]["team"]["points_per_shot"] == 1.0
    assert rows["rim"]["opp"] == {"shots": 1, "makes": 1, "points": 2, "points_per_shot": 2.0, "small_sample": True}


def test_fewer_than_twenty_shots_is_a_small_sample():
    rows = {r["zone"]: r for r in build_report(shots(), DEFAULT)["rows"]}
    assert rows["corner3"]["team"]["small_sample"] is False  # exactly 20
    assert rows["rim"]["team"]["small_sample"] is True       # 2


def test_margin_is_from_each_shooting_teams_side():
    leading_only = ClutchFilters.parse("2024-25", "TOR", "BOS", margin_min=1, margin_max=5)
    report = build_report(shots(), leading_only)
    assert report["totals"]["team"]["shots"] == 0  # TOR shot while trailing by 2
    assert report["totals"]["opp"]["shots"] == 1   # BOS shot while leading by 1


def test_no_matching_shots_explains_how_to_widen():
    report = build_report(shots(), ClutchFilters.parse("2022-23", "TOR", "BOS"))
    assert report["rows"] == [] and "wider margin" in report["message"]


def test_missing_columns_mean_data_unavailable():
    with pytest.raises(ClutchDataUnavailable):
        build_report(shots().drop(columns=["MARGIN"]), DEFAULT)


# Exports -----------------------------------------------------------------

def test_csv_has_one_row_per_zone_and_team_with_the_situation():
    report = build_report(shots(), DEFAULT)
    lines = report_csv(report).splitlines()
    assert "4th quarter and overtime, 5:00 or less left, margin -5 to +5" in lines[0]
    assert lines[2] == "zone,team,shots,makes,points,points_per_shot,small_sample"
    assert "Corner 3,TOR,20,10,30,1.500,no" in lines
    assert "Corner 3,BOS,0,0,0,,yes" in lines
    assert len(lines) == 3 + 2 * len(report["rows"])
    assert report_filename(report, "csv") == "clutch_TOR_vs_BOS_2024-25.csv"


def test_pdf_renders_with_and_without_rows():
    from infrastructure.visualization_and_export.clutch_report_pdf import render_clutch_pdf

    for filters in (DEFAULT, ClutchFilters.parse("2022-23", "TOR", "BOS")):
        report = build_report(shots(), filters)
        assert render_clutch_pdf(report, situation_text(report)).startswith(b"%PDF")


# API ---------------------------------------------------------------------

def client(loader=shots):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from application.api_coordination.clutch_report_endpoints import create_clutch_report_router

    app = FastAPI()
    app.include_router(create_clutch_report_router(loader))
    return TestClient(app)


def as_role(role):
    return patch(f"{SERVICE}.decode_supabase_jwt", return_value={"sub": "u1", "exp": 9999999999}), patch(
        f"{SERVICE}.get_profile_role", return_value=role
    )


AUTH = {"Authorization": "Bearer t"}
PARAMS = {"season": "2024-25", "team": "TOR", "opp": "BOS"}


def test_analyst_gets_the_report():
    a, b = as_role("analyst")
    with a, b:
        r = client().get("/data/clutch-report", params=PARAMS, headers=AUTH)
    assert r.status_code == 200 and r.json()["totals"]["team"]["shots"] == 22
    assert r.headers["cache-control"] == "private, no-store"


@pytest.mark.parametrize("role", ["coach", "pending"])
def test_other_roles_get_403(role):
    a, b = as_role(role)
    with a, b:
        assert client().get("/data/clutch-report", params=PARAMS, headers=AUTH).status_code == 403


def test_signed_out_gets_401():
    assert client().get("/data/clutch-report", params=PARAMS).status_code == 401


def test_bad_filter_is_400_and_missing_data_is_503():
    def missing():
        raise FileNotFoundError("shots parquet not found")

    a, b = as_role("analyst")
    with a, b:
        bad = client().get("/data/clutch-report", params={**PARAMS, "seconds_left": 900}, headers=AUTH)
        down = client(missing).get("/data/clutch-report", params=PARAMS, headers=AUTH)
    assert bad.status_code == 400 and "12 minutes" in bad.json()["detail"]
    assert down.status_code == 503 and down.json()["detail"] == "Shot data is not available yet."


def test_csv_and_pdf_download_with_filenames():
    a, b = as_role("analyst")
    with a, b:
        c = client()
        csv_r = c.get("/data/clutch-report.csv", params=PARAMS, headers=AUTH)
        pdf_r = c.get("/data/clutch-report.pdf", params=PARAMS, headers=AUTH)
    assert csv_r.headers["content-type"].startswith("text/csv")
    assert 'filename="clutch_TOR_vs_BOS_2024-25.csv"' in csv_r.headers["content-disposition"]
    assert pdf_r.headers["content-type"] == "application/pdf" and pdf_r.content.startswith(b"%PDF")
