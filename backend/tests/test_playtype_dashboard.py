"""Independent source arithmetic and coach access checks for the dashboard."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from domain.baseline_recommendation.playtype_dashboard import PlaytypeDashboardService

AUTH = "application.access_control_services.access_control_service"


def request(params=None, *, role="coach", token=True):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "dashboard-test"}), patch(
        f"{AUTH}.get_profile_role", return_value=role
    ), patch("application.api_coordination.request_log_endpoints.store.submit_record"):
        return TestClient(app_module.app).get(
            "/analytics/playtypes", params=params or {"season": "2024-25", "our": "TOR", "opp": "BOS"},
            headers={"Authorization": "Bearer test"} if token else {},
        )


@pytest.mark.parametrize("season,our,opp", [("2024-25", "TOR", "BOS"), ("2019-20", "LAL", "SAS")])
def test_dashboard_matches_independent_regular_source_totals(season, our, opp):
    result = request({"season": season, "our": our, "opp": opp})
    assert result.status_code == 200
    body = result.json()
    raw = app_module.rec.raw_df
    regular = raw.loc[raw.SEASON.eq(season) & raw.SEASON_ID.astype(str).eq("2" + season[:4])]
    offense = regular.loc[regular.TYPE_GROUPING.eq("Offensive")]
    our_source = offense.loc[offense.TEAM_ABBREVIATION.eq(our)]
    defense = regular.loc[regular.TYPE_GROUPING.eq("Defensive")]
    opponent = defense.loc[defense.TEAM_ABBREVIATION.eq(opp)]
    assert len(body["usage_efficiency"]["rows"]) == 11
    assert len(body["opponent_defense"]["rows"]) == 7
    assert body["coverage"]["unavailable_defense_types"] == ["Cut", "Misc", "OffRebound", "Transition"]
    assert body["usage_efficiency"]["total_possessions"] == our_source.POSS.sum()
    assert body["usage_efficiency"]["league_total_possessions"] == offense.POSS.sum()
    assert body["opponent_defense"]["tracked_possessions"] == opponent.POSS.sum()
    for row in body["usage_efficiency"]["rows"]:
        selected = our_source.loc[our_source.PLAY_TYPE.eq(row["play_type"])]
        benchmark = offense.loc[offense.PLAY_TYPE.eq(row["play_type"])]
        assert row["possessions"] == selected.POSS.sum()
        assert row["points"] == selected.PTS.sum()
        assert row["ppp"] == pytest.approx(selected.PTS.sum() / selected.POSS.sum())
        assert row["usage_pct"] == pytest.approx(100 * selected.POSS.sum() / our_source.POSS.sum())
        assert row["league_ppp"] == pytest.approx(benchmark.PTS.sum() / benchmark.POSS.sum())
        assert row["league_usage_pct"] == pytest.approx(100 * benchmark.POSS.sum() / offense.POSS.sum())
        assert row["usage_delta_pp"] == pytest.approx(row["usage_pct"] - row["league_usage_pct"])
        assert row["ppp_delta"] == pytest.approx(row["ppp"] - row["league_ppp"])
    assert sum(row["usage_pct"] for row in body["usage_efficiency"]["rows"]) == pytest.approx(100)
    for row in body["opponent_defense"]["rows"]:
        selected = opponent.loc[opponent.PLAY_TYPE.eq(row["play_type"])]
        benchmark = defense.loc[defense.PLAY_TYPE.eq(row["play_type"])]
        assert row["possessions"] == selected.POSS.sum()
        assert row["points_allowed"] == selected.PTS.sum()
        assert row["allowed_ppp"] == pytest.approx(selected.PTS.sum() / selected.POSS.sum())
        assert row["league_allowed_ppp"] == pytest.approx(benchmark.PTS.sum() / benchmark.POSS.sum())
        assert row["allowed_ppp_delta"] == pytest.approx(row["allowed_ppp"] - row["league_allowed_ppp"])
    assert body["filters"]["season_phase"] == "regular_season"
    assert body["aggregation_version"] == "synergy_regular_season_v2"
    assert body["dataset_versions"]["synergy"] == app_module.rec.raw_df.attrs["dataset_version"]
    assert result.headers["X-Dataset-Version"] == body["dataset_versions"]["synergy"]
    assert result.headers["X-Aggregation-Version"] == body["aggregation_version"]
    assert result.headers["Cache-Control"] == "private, no-store"


def test_cutoff_marks_all_rows_without_hiding_or_inventing_a_top_three():
    low = request().json()
    high = request({"season": "2024-25", "our": "TOR", "opp": "BOS", "min_poss": 10000}).json()
    for section in ["usage_efficiency", "opponent_defense"]:
        assert [row["play_type"] for row in high[section]["rows"]] == [row["play_type"] for row in low[section]["rows"]]
        assert all(not row["sample_ok"] for row in high[section]["rows"])
    assert all(row["weak_spot_rank"] is None for row in high["opponent_defense"]["rows"])


def test_relative_defense_ranking_uses_same_play_type_benchmark_and_support():
    offense = [{"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": play,
                "POSS": 100, "PTS": 100, "PPP": 1.0} for play in "ABCD"]
    defense = [{"SEASON": "2024-25", "TEAM_ABBREVIATION": "BOS", "SIDE": "defense", "PLAY_TYPE": play,
                "POSS": poss, "PTS": ppp * poss, "PPP": ppp}
               for play, poss, ppp in [("A", 100, 1.3), ("B", 100, 1.2), ("C", 19, 1.0), ("D", 100, 0.9)]]
    league = pd.DataFrame([
        *[{"SEASON": "2024-25", "SIDE": "offense", "PLAY_TYPE": play, "LEAGUE_POSS": 1000, "PPP": 1.0} for play in "ABCD"],
        *[{"SEASON": "2024-25", "SIDE": "defense", "PLAY_TYPE": play, "LEAGUE_POSS": 1000, "PPP": ppp}
          for play, ppp in [("A", 1.4), ("B", 0.8), ("C", 0.9), ("D", 0.9)]],
    ])
    service = PlaytypeDashboardService(pd.DataFrame(offense + defense), league, "sha256:" + "a" * 64)
    rows = service.build("2024-25", "TOR", "BOS")["opponent_defense"]["rows"]
    assert [row["play_type"] for row in rows] == ["B", "C", "D", "A"]
    assert [row["weak_spot_rank"] for row in rows] == [1, None, None, None]
    assert rows[1]["sample_ok"] is False


@pytest.mark.parametrize("params,status", [
    ({"season": "1900-01", "our": "TOR", "opp": "BOS"}, 400),
    ({"season": "2024-25", "our": "INVALID", "opp": "BOS"}, 400),
    ({"season": "2024-25", "our": "TOR", "opp": "TOR"}, 400),
    ({"season": "2024-25", "our": "TOR", "opp": "BOS", "min_poss": 0}, 422),
    ({"season": "2024-25", "our": "TOR", "opp": "BOS", "min_poss": 20.5}, 422),
    ({"season": "2024-25", "our": "TOR", "opp": "BOS", "min_poss": 10001}, 422),
])
def test_invalid_inputs_return_errors_without_chart_rows(params, status):
    result = request(params)
    assert result.status_code == status
    assert "usage_efficiency" not in result.json()


@pytest.mark.parametrize("role,token,status", [("coach", False, 401), ("analyst", True, 403), ("pending", True, 403)])
def test_dashboard_enforces_current_coach_role(role, token, status):
    result = request(role=role, token=token)
    assert result.status_code == status
    assert "usage_efficiency" not in result.json()


def test_quality_failure_blocks_dashboard(monkeypatch):
    monkeypatch.setattr(app_module.data_quality_gate, "report", lambda: {"status": "fail", "checks": [{"status": "fail", "name": "fixture"}]})
    result = request()
    assert result.status_code == 503
    assert "usage_efficiency" not in result.json()


def test_missing_source_is_unavailable_not_empty_success(monkeypatch):
    monkeypatch.setattr(app_module, "rec", None)
    assert request().status_code == 503
