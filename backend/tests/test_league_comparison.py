"""Independent source totals and missing-denominator checks for league rates."""

from types import SimpleNamespace

import pandas as pd
import pytest

from application.analytics_services.league_comparison import build_league_comparison, LeagueDataUnavailable
from application.api_coordination.app import rec


@pytest.mark.parametrize("side,group", [("offense", "Offensive"), ("defense", "Defensive")])
def test_comparison_matches_raw_player_totals(side, group):
    result = build_league_comparison(rec, "2024-25", "TOR", side)
    raw = rec.raw_df
    source = raw.loc[raw.SEASON_ID.astype(str).eq("22024") & raw.TYPE_GROUPING.eq(group)]
    assert result["selected_team_in_league"] is True
    assert result["dataset_version"] == raw.attrs["dataset_version"]
    assert {row["play_type"] for row in result["rows"]} == set(source.PLAY_TYPE)
    for row in result["rows"]:
        category = source.loc[source.PLAY_TYPE.eq(row["play_type"])]
        for key, frame in [("team", category.loc[category.TEAM_ABBREVIATION.eq("TOR")]), ("league", category)]:
            value = row[key]
            assert value["possessions"] == frame.POSS.sum()
            assert value["field_goal_attempts"] == frame.FGA.sum()
            assert value["fg_pct"] == pytest.approx(frame.FGM.sum() / frame.FGA.sum())
            assert value["efg_pct"] == pytest.approx((frame.EFG_PCT * frame.FGA).sum() / frame.FGA.sum())
            assert value["tov_pct"] == pytest.approx((frame.TOV_POSS_PCT * frame.POSS).sum() / frame.POSS.sum())
            assert value["ft_pct"] == pytest.approx((frame.FT_POSS_PCT * frame.POSS).sum() / frame.POSS.sum())
        for rate, difference in row["difference_pp"].items():
            assert difference == pytest.approx(100 * (row["team"][rate] - row["league"][rate]))


def fixture_rec(**values):
    team = {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "Cut",
            "POSS": 10, "FGA": 0, "FG_PCT": float("nan"), "EFG_PCT": float("nan"),
            "TOV_POSS_PCT": 0.2, "FT_POSS_PCT": 0.4, **values}
    league = {**team, "LEAGUE_POSS": 100, "FGA": 80, "FG_PCT": 0.5, "EFG_PCT": 0.6}
    return SimpleNamespace(team_df=pd.DataFrame([team]), league_df=pd.DataFrame([league]), raw_df=pd.DataFrame())


def test_zero_attempts_keep_possession_rates():
    row = build_league_comparison(fixture_rec(), "2024-25", "TOR")["rows"][0]
    assert row["team"]["fg_pct"] is None and row["team"]["efg_pct"] is None
    assert row["difference_pp"]["fg_pct"] is None
    assert row["team"]["tov_pct"] == 0.2 and row["team"]["ft_pct"] == 0.4
    assert row["team"]["possessions"] == 10 and row["team"]["field_goal_attempts"] == 0


def test_missing_categories_are_not_invented_zeros():
    data = fixture_rec()
    data.league_df.loc[0, "PLAY_TYPE"] = "Spotup"
    cut, spotup = build_league_comparison(data, "2024-25", "TOR")["rows"]
    assert cut["league"] is None and spotup["team"] is None
    assert all(v is None for v in cut["difference_pp"].values())
    missing = build_league_comparison(data, "2023-24", "TOR")
    assert missing["status"] == "no_data" and missing["rows"] == []


@pytest.mark.parametrize("values", [{"POSS": -1}, {"FGA": 0.5}, {"TOV_POSS_PCT": float("nan")},
    {"FT_POSS_PCT": 1.1}, {"POSS": None}, {"TOV_POSS_PCT": "broken"}])
def test_bad_source_values_are_errors(values):
    with pytest.raises(LeagueDataUnavailable):
        build_league_comparison(fixture_rec(**values), "2024-25", "TOR")


@pytest.mark.parametrize("args", [("2024-99", "TOR", "offense"), ("2024-25", "XXX", "offense"), ("2024-25", "TOR", "both")])
def test_invalid_filters(args):
    with pytest.raises(ValueError):
        build_league_comparison(rec, *args)
