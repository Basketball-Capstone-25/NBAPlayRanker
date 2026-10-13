"""Generic /export/report endpoint on the unified export contract (SCRUM-485)."""

from __future__ import annotations

import csv
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module

AUTH_SERVICE = "application.access_control_services.access_control_service"
HEADERS = {"Authorization": "Bearer test"}
SEASON = app_module.VALID_SEASONS[-1]
MATCHUP = {"season": SEASON, "our": "TOR", "opp": "BOS", "k": 3, "w_off": 0.7}


def _get(path, params, role="coach", headers=HEADERS):
    client = TestClient(app_module.app)
    with patch(f"{AUTH_SERVICE}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH_SERVICE}.get_profile_role", return_value=role,
    ):
        return client.get(path, params=params, headers=headers)


def test_baseline_csv_matches_the_ranking_page():
    page = _get("/rank-plays/baseline", MATCHUP).json()["rankings"]
    res = _get("/export/report", {"report": "baseline_rankings", "format": "csv", **MATCHUP})

    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")
    assert res.headers["content-disposition"] == (
        f'attachment; filename="baseline-rankings_{SEASON}_TOR_BOS_3_0.7.csv"'
    )
    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert [row["PLAY_TYPE"] for row in rows] == [row["PLAY_TYPE"] for row in page]
    assert list(rows[0]) == list(page[0])


def test_json_carries_the_same_rows_as_csv():
    csv_rows = list(csv.DictReader(io.StringIO(
        _get("/export/report", {"report": "baseline_rankings", "format": "csv", **MATCHUP}).text
    )))
    body = _get("/export/report", {"report": "baseline_rankings", "format": "json", **MATCHUP}).json()

    assert body["report"] == "baseline_rankings"
    assert body["filters"]["our"] == "TOR"
    assert [row["PLAY_TYPE"] for row in body["rows"]] == [row["PLAY_TYPE"] for row in csv_rows]


@pytest.mark.parametrize("report,role,params", [
    ("baseline_rankings", "coach", MATCHUP),
    ("team_playtypes", "analyst", {"season": SEASON, "side": "offense"}),
    ("topk_uplift", "analyst", MATCHUP),
])
@pytest.mark.parametrize("fmt,media", [("csv", "text/csv"), ("json", "application/json"), ("pdf", "application/pdf")])
def test_every_report_exports_in_every_format(report, role, params, fmt, media):
    res = _get("/export/report", {"report": report, "format": fmt, **params}, role=role)

    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith(media)
    assert res.headers["content-disposition"].endswith(f'.{fmt}"')
    if fmt == "pdf":
        assert res.content.startswith(b"%PDF")


def test_topk_uplift_export_includes_the_summary_figures():
    body = _get("/export/report", {"report": "topk_uplift", "format": "json", **MATCHUP}, role="analyst").json()

    assert {"team_season_ppp", "topk_modeled_ppp", "uplift_ppp"} <= set(body["metadata"])
    assert len(body["rows"]) == 3


@pytest.mark.parametrize("params", [
    {"report": "nope", "format": "csv", **MATCHUP},
    {"report": "baseline_rankings", "format": "xlsx", **MATCHUP},
    {"report": "baseline_rankings", "format": "csv", **{**MATCHUP, "opp": "TOR"}},
    {"report": "baseline_rankings", "format": "csv", **{**MATCHUP, "season": "1999-00"}},
    {"report": "team_playtypes", "format": "csv", "season": SEASON, "side": "middle"},
])
def test_invalid_requests_return_400(params):
    role = "analyst" if params["report"] == "team_playtypes" else "coach"
    assert _get("/export/report", params, role=role).status_code == 400


@pytest.mark.parametrize("report,role", [
    ("baseline_rankings", "analyst"),
    ("team_playtypes", "coach"),
    ("topk_uplift", "coach"),
])
def test_each_report_keeps_its_role_check(report, role):
    params = {"report": report, "format": "csv", **MATCHUP, "side": "offense"}
    assert _get("/export/report", params, role=role).status_code == 403


def test_missing_session_is_rejected():
    res = _get("/export/report", {"report": "baseline_rankings", "format": "csv", **MATCHUP}, headers={})
    assert res.status_code == 401
