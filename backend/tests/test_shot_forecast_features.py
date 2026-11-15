"""Only completed earlier dates may supply pre-game shot predictors."""

import numpy as np
import pandas as pd
import pytest

from domain.shot_analysis.shot_forecast_features import FEATURE_COLUMNS, PROOF_COLUMNS, build_pregame_rows


def game(game_id, date, season, home, away, home_points, away_points):
    rows = []
    for team, opponent, is_home, points in [(home, away, True, home_points), (away, home, False, away_points)]:
        for index, outcome in enumerate(points):
            rows.append({
                "EVENT_ID": f"{game_id}-{team}-{index}", "GAME_ID": game_id, "SEASON_STR": season,
                "TEAM_ABBR": team, "OPP_ABBR": opponent, "HOME_FLAG": is_home,
                "POINTS": outcome, "GAME_DATE": pd.Timestamp(date),
            })
    return rows


@pytest.fixture
def shots():
    return pd.DataFrame([
        *game("g1", "2024-04-01", "2023-24", "A", "B", [0, 2, 3], [0, 2]),
        *game("g2", "2024-04-02", "2023-24", "C", "A", [0, 3, 3], [0, 0]),
        *game("g3", "2024-04-02", "2023-24", "A", "D", [2], [0]),
        *game("g4", "2024-10-22", "2024-25", "A", "B", [0, 2], [3]),
    ])


def proof(rows):
    return rows[["GAME_ID", "TEAM_ABBR", *FEATURE_COLUMNS, *PROOF_COLUMNS]].reset_index(drop=True)


def test_weighted_prior_totals_outcomes_and_cross_season_history(shots):
    unchanged = shots.copy(deep=True)
    rows, coverage = build_pregame_rows(shots)
    final = rows.query("GAME_ID == 'g4' and TEAM_ABBR == 'A'").iloc[0]
    assert final.PRIOR_TEAM_PPS == pytest.approx(7 / 6)
    assert final.PRIOR_OPPONENT_ALLOWED_PPS == pytest.approx(5 / 3)
    assert final.PRIOR_LEAGUE_PPS == pytest.approx(15 / 12)
    assert (final.PRIOR_TEAM_FGA, final.PRIOR_TEAM_POINTS) == (6, 7)
    assert (final.PRIOR_OPPONENT_ALLOWED_FGA, final.PRIOR_OPPONENT_ALLOWED_POINTS) == (3, 5)
    assert (final.PRIOR_LEAGUE_FGA, final.PRIOR_LEAGUE_POINTS) == (12, 15)
    assert final.HISTORY_CUTOFF_DATE == final.PRIOR_TEAM_LAST_DATE == pd.Timestamp("2024-04-02")
    assert final.PRIOR_OPPONENT_ALLOWED_LAST_DATE == pd.Timestamp("2024-04-01")
    assert (final.FGA, final.POINTS, final.ZERO_POINT_SHOTS, final.TWO_POINT_SHOTS, final.THREE_POINT_SHOTS) == (2, 2, 1, 1, 0)
    assert final.TARGET_PPS == 1
    assert coverage["input_shots"] == 15 and coverage["scored_shots"] == 10
    assert coverage["input_games"] == 4 and coverage["scored_game_team_rows"] == 6
    assert coverage["warmup_dates"] == ["2024-04-01"] and coverage["warmup_shots"] == 5
    assert coverage["by_season"][1]["warmup_shots"] == 0
    assert (rows.HISTORY_CUTOFF_DATE < rows.GAME_DATE).all()
    pd.testing.assert_frame_equal(shots, unchanged)


def test_missing_histories_use_available_league_average_with_zero_raw_counts(shots):
    rows, coverage = build_pregame_rows(shots)
    newcomer = rows.query("GAME_ID == 'g2' and TEAM_ABBR == 'C'").iloc[0]
    assert newcomer.PRIOR_TEAM_PPS == newcomer.PRIOR_LEAGUE_PPS == 1.4
    assert newcomer.TEAM_HISTORY_FALLBACK and not newcomer.OPPONENT_HISTORY_FALLBACK
    assert newcomer.PRIOR_TEAM_FGA == newcomer.PRIOR_TEAM_POINTS == 0
    assert pd.isna(newcomer.PRIOR_TEAM_LAST_DATE)
    assert newcomer.PRIOR_OPPONENT_ALLOWED_PPS == 1
    against_newcomer = rows.query("GAME_ID == 'g3' and TEAM_ABBR == 'A'").iloc[0]
    assert against_newcomer.OPPONENT_HISTORY_FALLBACK
    assert against_newcomer.PRIOR_OPPONENT_ALLOWED_FGA == 0
    assert against_newcomer.PRIOR_OPPONENT_ALLOWED_PPS == 1.4
    assert coverage["team_fallback_rows"] == coverage["opponent_fallback_rows"] == 2


def test_current_date_outcomes_cannot_change_same_date_predictions(shots):
    original, _ = build_pregame_rows(shots)
    changed = shots.copy()
    changed.loc[changed.GAME_ID.eq("g2"), "POINTS"] = 3
    later, _ = build_pregame_rows(changed)
    same_date = original.GAME_DATE.eq(pd.Timestamp("2024-04-02"))
    pd.testing.assert_frame_equal(proof(original[same_date]), proof(later[same_date]))
    # Once that date has passed, its observed results may update future features.
    assert original.iloc[-1].PRIOR_LEAGUE_PPS != later.iloc[-1].PRIOR_LEAGUE_PPS


def test_future_outcomes_and_observed_shot_fields_do_not_change_any_features(shots):
    original, _ = build_pregame_rows(shots)
    changed = shots.assign(X=999, Y=-999, SHOT_TYPE="changed", MARGIN=99, CLOCK_SEC=0, MADE=True)
    changed.loc[changed.GAME_ID.eq("g4"), "POINTS"] = 3
    later, _ = build_pregame_rows(changed)
    pd.testing.assert_frame_equal(proof(original), proof(later))
    assert not original.TARGET_PPS.equals(later.TARGET_PPS)


def test_new_same_date_game_cannot_supply_history_for_other_games_that_day(shots):
    original, _ = build_pregame_rows(shots)
    added = pd.DataFrame(game("g0", "2024-04-02", "2023-24", "A", "B", [3, 3], [3, 3]))
    changed, _ = build_pregame_rows(pd.concat([shots, added], ignore_index=True))
    unchanged_day = changed.GAME_ID.isin(["g2", "g3"])
    pd.testing.assert_frame_equal(proof(original[original.GAME_ID.isin(["g2", "g3"])]), proof(changed[unchanged_day]))


def test_row_order_cannot_change_features_or_coverage(shots):
    expected, coverage = build_pregame_rows(shots)
    actual, shuffled_coverage = build_pregame_rows(shots.sample(frac=1, random_state=31))
    pd.testing.assert_frame_equal(expected, actual)
    assert shuffled_coverage == coverage


def test_only_warmup_returns_empty_rows_and_counts_every_exclusion(shots):
    rows, coverage = build_pregame_rows(shots[shots.GAME_ID.eq("g1")])
    assert rows.empty and set(FEATURE_COLUMNS).issubset(rows.columns)
    assert coverage["scored_shots"] == coverage["scored_game_team_rows"] == 0
    assert coverage["warmup_shots"] == 5 and coverage["warmup_game_team_rows"] == 2


@pytest.mark.parametrize("column,value", [
    ("POINTS", -1), ("POINTS", 1), ("POINTS", 4), ("POINTS", np.nan), ("POINTS", np.inf), ("POINTS", True),
    ("EVENT_ID", None), ("GAME_ID", ""), ("TEAM_ABBR", ""), ("OPP_ABBR", None),
    ("SEASON_STR", "2024"), ("GAME_DATE", pd.NaT), ("GAME_DATE", "invalid"),
    ("GAME_DATE", pd.Timestamp("2024-04-01 23:00")), ("HOME_FLAG", "False"), ("HOME_FLAG", 2),
])
def test_invalid_prepared_values_are_rejected(shots, column, value):
    changed = shots.astype({column: object})
    changed.loc[0, column] = value
    with pytest.raises(ValueError):
        build_pregame_rows(changed)


@pytest.mark.parametrize("column,value", [
    ("EVENT_ID", "g1-A-1"), ("OPP_ABBR", "A"), ("OPP_ABBR", "C"), ("HOME_FLAG", False),
    ("GAME_DATE", pd.Timestamp("2024-04-02")), ("SEASON_STR", "2024-25"),
])
def test_duplicate_events_and_conflicting_game_context_are_rejected(shots, column, value):
    changed = shots.copy()
    changed.loc[0, column] = value
    with pytest.raises(ValueError):
        build_pregame_rows(changed)


def test_timezone_dates_are_rejected_instead_of_silently_changing_cutoffs(shots):
    changed = shots.assign(GAME_DATE=shots.GAME_DATE.dt.tz_localize("America/New_York"))
    with pytest.raises(ValueError, match="calendar date"):
        build_pregame_rows(changed)


def test_missing_fields_and_empty_input_are_rejected(shots):
    for changed in [shots.iloc[:0], shots.drop(columns="EVENT_ID")]:
        with pytest.raises(ValueError, match="Prepared shots"):
            build_pregame_rows(changed)
