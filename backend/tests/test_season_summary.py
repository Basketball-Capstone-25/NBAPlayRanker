"""Season denominators, missing periods and shared-role API access."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.analytics_services.season_summary import build_season_summary
from application.api_coordination import app as app_module

AUTH = "application.access_control_services.access_control_service"
PATH = "/analytics/playtypes/summary"
PARAMS = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 3}


def request(params=None, *, role="coach", token=True):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "summary-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        return TestClient(app_module.app).get(
            PATH, params=PARAMS if params is None else params,
            headers={"Authorization": "Bearer test"} if token else {},
        )


@pytest.mark.parametrize("role", ["coach", "analyst"])
@pytest.mark.parametrize("w_off", [0.0, 0.7, 1.0])
def test_summary_matches_raw_season_totals_and_existing_uplift(role, w_off):
    params = {**PARAMS, "w_off": w_off}
    response = request(params, role=role)
    assert response.status_code == 200
    body = response.json()
    raw = app_module.rec.raw_df
    source = raw.loc[
        raw.SEASON.eq("2024-25") & raw.SEASON_ID.astype(str).eq("22024")
        & raw.TEAM_ABBREVIATION.eq("TOR") & raw.TYPE_GROUPING.eq("Offensive")
    ]
    summary = body["summary"]
    assert body["status"] == "ok"
    assert summary["team_possessions"] == source.POSS.sum()
    assert summary["observed_team_ppp"] == pytest.approx(source.PTS.sum() / source.POSS.sum())
    for row in body["play_types"]:
        poss = source.loc[source.PLAY_TYPE.eq(row["play_type"]), "POSS"].sum()
        assert row["possessions"] == poss
        assert row["usage_pct"] == pytest.approx(100 * poss / source.POSS.sum())
    assert summary["selected_usage_pct"] == pytest.approx(sum(r["usage_pct"] for r in body["play_types"]))
    assert 0 < summary["selected_usage_pct"] < 100
    assert summary["k_returned"] == 3
    assert summary["modeled_uplift_ppp"] == pytest.approx(summary["modeled_topk_ppp"] - summary["observed_team_ppp"])
    assert body["dataset_version"] == app_module.rec.raw_df.attrs["dataset_version"]
    assert body["filters"]["season_phase"] == "regular_season"
    assert body["filters"]["w_off"] == w_off
    assert body["filters"]["w_def"] == 1 - w_off
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Data-Quality"] == "pass"
    # Compare with the existing analyst evidence API, not a second formula.
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "summary-test"}), patch(
        f"{AUTH}.get_profile_role", return_value="analyst"
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        evidence = TestClient(app_module.app).get(
            "/metrics/topk-uplift", params=params, headers={"Authorization": "Bearer test"},
        ).json()
    assert summary["modeled_topk_ppp"] == evidence["topk"]["modeled_ppp"]
    assert summary["modeled_uplift_ppp"] == evidence["uplift_ppp"]


def test_shortlist_usage_does_not_become_100_percent(monkeypatch):
    team = pd.DataFrame([
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "Cut", "PTS": 80, "POSS": 100},
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "Transition", "PTS": 600, "POSS": 300},
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "BOS", "SIDE": "defense", "PLAY_TYPE": "Cut", "PTS": 90, "POSS": 100},
    ])
    rec = SimpleNamespace(team_df=team, league_df=pd.DataFrame(), raw_df=pd.DataFrame())
    monkeypatch.setattr("application.analytics_services.season_summary.rank_playtypes_baseline", lambda *a, **kw:
                        pd.DataFrame([{"PLAY_TYPE": "Cut", "PPP_PRED": 0.9, "POSS_OFF": 100}]))
    body = build_season_summary(rec, "2024-25", "TOR", "BOS", 1)
    assert body["play_types"][0]["usage_pct"] == 25
    assert body["summary"]["selected_usage_pct"] == 25
    assert body["summary"]["observed_team_ppp"] == 1.7
    assert body["summary"]["modeled_uplift_ppp"] == pytest.approx(-0.8)


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("pending", True, 403), (None, True, 403)])
def test_summary_requires_a_current_assigned_role(role, token, status):
    assert request(role=role, token=token).status_code == status


@pytest.mark.parametrize("change,status", [
    ({"k": 0}, 422), ({"k": 11}, 422), ({"k": 1.5}, 422),
    ({"period": "last10"}, 422), ({"season": "2024-99"}, 400),
    ({"our": "XXX"}, 400), ({"opp": "TOR"}, 400),
    ({"w_off": "nan"}, 422), ({"w_off": -0.1}, 422), ({"w_off": 1.1}, 422),
])
def test_invalid_or_unsupported_filters_are_not_silently_accepted(change, status):
    assert request({**PARAMS, **change}).status_code == status


@pytest.mark.parametrize("missing", ["season", "opponent", "matching_types"])
def test_missing_period_or_matchup_has_no_invented_zero_summary(monkeypatch, missing):
    params = dict(PARAMS)
    frame = app_module.rec.team_df.copy(deep=True)
    if missing == "season":
        params["season"] = "2028-29"
    else:
        selected = frame.SEASON.eq("2024-25") & frame.TEAM_ABBREVIATION.eq("BOS") & frame.SIDE.eq("defense")
        if missing == "opponent":
            frame = frame.loc[~selected]
        else:
            frame.loc[selected, "PLAY_TYPE"] = "Unmatched"
        monkeypatch.setattr(app_module.rec, "team_df", frame)
    body = request(params).json()
    assert body["status"] == "no_data"
    assert body["message"] == "No data for this period."
    assert body["summary"] is None
    assert body["play_types"] == []


def test_bad_data_is_an_error_not_an_empty_period(monkeypatch):
    frame = app_module.rec.team_df.copy(deep=True)
    frame.loc[frame.TEAM_ABBREVIATION.eq("TOR"), "PTS"] = float("nan")
    monkeypatch.setattr(app_module.rec, "team_df", frame)
    response = request()
    assert response.status_code == 503
    assert "summary" not in response.json()


def test_quality_gate_also_protects_summary(monkeypatch):
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {
        "status": "fail", "checks": [{"status": "fail", "name": "fixture"}],
    })
    response = request()
    assert response.status_code == 503
    assert response.headers["X-Data-Quality"] == "fail"
