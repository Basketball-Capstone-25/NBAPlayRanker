"""Arithmetic and source-scope regressions for the play-type baseline."""

import pandas as pd
import pytest
from types import SimpleNamespace

from domain.baseline_recommendation.baseline_recommender import (
    WEIGHT_COLS,
    build_league_averages,
    build_team_playtype_tables,
    rank_playtypes_baseline,
)


def player_row(player, *, poss, points, fga, fgm, efg, season_id=22024, team="TOR"):
    return {
        "SEASON": "2024-25", "SEASON_ID": season_id,
        "ENTITY_TYPE": "Player", "PLAYER_ID": player,
        "TEAM_ABBREVIATION": team, "TEAM_NAME": team,
        "PLAY_TYPE": "Isolation", "TYPE_GROUPING": "Offensive", "GP": 10,
        "POSS": poss, "PTS": points, "FGA": fga, "FGM": fgm,
        "PPP": round(points / poss, 3), "FG_PCT": round(fgm / fga, 3),
        "EFG_PCT": efg, "POSS_PCT": 0.2, "SCORE_POSS_PCT": 0.5,
        "TOV_POSS_PCT": 0.1, "SF_POSS_PCT": 0.1,
        "FT_POSS_PCT": 0.1, "PLUSONE_POSS_PCT": 0.02,
    }


@pytest.fixture
def source():
    return pd.DataFrame([
        player_row(1, poss=100, points=90, fga=10, fgm=5, efg=0.6),
        player_row(2, poss=10, points=20, fga=10, fgm=10, efg=1.0),
    ])


def test_regular_season_totals_exclude_playoffs(source):
    playoff = player_row(1, poss=50, points=100, fga=50, fgm=50, efg=1.0, season_id=42024)
    row = build_team_playtype_tables(pd.concat([source, pd.DataFrame([playoff])])).iloc[0]
    assert row["POSS"] == 110
    assert row["PTS"] == 110
    assert row["PPP"] == pytest.approx(1.0)


def test_field_goal_rates_use_attempts_and_ppp_uses_exact_counts(source):
    source.loc[0, "PPP"] = 0.899  # Stored rounded rates must not replace exact counts.
    row = build_team_playtype_tables(source).iloc[0]
    assert row["PPP"] == pytest.approx(1.0)
    assert row["FG_PCT"] == pytest.approx(15 / 20)
    assert row["EFG_PCT"] == pytest.approx((0.6 * 10 + 1.0 * 10) / 20)


def test_league_rates_use_counts_and_their_own_denominators(source):
    other = player_row(3, poss=10, points=5, fga=10, fgm=1, efg=0.1, team="BOS")
    teams = build_team_playtype_tables(pd.concat([source, pd.DataFrame([other])]))
    row = build_league_averages(teams).iloc[0]
    assert row["PPP"] == pytest.approx(115 / 120)
    assert row["FG_PCT"] == pytest.approx(16 / 30)
    assert row["EFG_PCT"] == pytest.approx(17 / 30)


def test_duplicate_player_source_row_cannot_double_count_possessions(source):
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        build_team_playtype_tables(pd.concat([source, source.iloc[[0]]]))


def test_source_phase_is_required_instead_of_silently_mixing_seasons(source):
    with pytest.raises(ValueError, match="SEASON_ID"):
        build_team_playtype_tables(source.drop(columns="SEASON_ID"))


@pytest.mark.parametrize("column,value", [("POSS", 0), ("POSS", 0.5), ("FGM", 11), ("FT_POSS_PCT", -0.1), ("EFG_PCT", float("nan"))])
def test_invalid_source_values_are_rejected(source, column, value):
    source[column] = source[column].astype(float)
    source.loc[0, column] = value
    with pytest.raises(ValueError):
        build_team_playtype_tables(source)


def test_weaker_opponent_defense_does_not_lower_offensive_score():
    common = {
        "SEASON": "2024-25", "PLAY_TYPE": "Isolation", "POSS": 100,
        "POSS_PCT": 0.1, "RELIABILITY_WEIGHT": 1.0,
        **{column: 0.2 for column in WEIGHT_COLS},
    }
    teams = pd.DataFrame([
        {**common, "SIDE": "offense", "TEAM_ABBREVIATION": "TOR", "PPP": 1.1},
        {**common, "SIDE": "defense", "TEAM_ABBREVIATION": "BOS", "PPP": 0.9},
    ])
    league = pd.DataFrame([
        {"SEASON": "2024-25", "PLAY_TYPE": "Isolation", "SIDE": side, "PPP": 1.0}
        for side in ["offense", "defense"]
    ])
    stronger = rank_playtypes_baseline(teams, league, "2024-25", "TOR", "BOS", k=1)
    teams.loc[teams.SIDE.eq("defense"), "PPP"] = 1.1
    weaker = rank_playtypes_baseline(teams, league, "2024-25", "TOR", "BOS", k=1)
    assert weaker.iloc[0]["PPP_PRED"] > stronger.iloc[0]["PPP_PRED"]
    assert weaker.iloc[0]["PPP_PRED"] == pytest.approx(1.1)


def test_context_blend_keeps_the_same_defense_direction(monkeypatch):
    from domain.context_ml_recommendation import ml_context_recommender as context
    common = {"SEASON": "2024-25", "PLAY_TYPE": "Isolation", "POSS": 100,
              "POSS_PCT": 0.1, "RELIABILITY_WEIGHT": 1.0, "EFG_PCT": 0.5, "TOV_POSS_PCT": 0.1}
    team = pd.DataFrame([
        {**common, "SIDE": "offense", "TEAM_ABBREVIATION": "TOR", "PPP": 1.1},
        {**common, "SIDE": "defense", "TEAM_ABBREVIATION": "BOS", "PPP": 0.9},
    ])
    league = pd.DataFrame([{"SEASON": "2024-25", "PLAY_TYPE": "Isolation", "SIDE": side, "PPP": 1.0}
                           for side in ["offense", "defense"]])
    predictions = pd.DataFrame([{"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "PLAY_TYPE": "Isolation", "PPP_ML": 1.05}])
    monkeypatch.setattr(context, "_rec", SimpleNamespace(team_df=team, league_df=league, raw_df=pd.DataFrame()))
    monkeypatch.setattr(context, "_get_ml_pred_df", lambda: predictions)
    strong = context.build_ml_matchup_table("2024-25", "TOR", "BOS")
    team.loc[team.SIDE.eq("defense"), "PPP"] = 1.1
    weak = context.build_ml_matchup_table("2024-25", "TOR", "BOS")
    for column in ["PPP_BASELINE", "PPP_ML_BLEND"]:
        assert weak.iloc[0][column] - strong.iloc[0][column] == pytest.approx(0.06)
