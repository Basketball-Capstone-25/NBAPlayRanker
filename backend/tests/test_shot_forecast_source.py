"""Forecast source joins preserve attempts and reject ambiguous event history."""

from datetime import date

import pandas as pd
import pytest

from domain.shot_analysis.shot_forecast_source import SHOT_COLUMNS, prepare_shot_forecast_rows


def source():
    shots = pd.DataFrame([
        {"EVENT_ID": "1011", "GAME_ID": "101", "SEASON_STR": "2023-24", "TEAM_ABBR": "TOR",
         "OPP_ABBR": "BOS", "HOME_FLAG": True, "POINTS": 2, "X": 5, "LOCATION_VALID": False},
        {"EVENT_ID": "1012", "GAME_ID": "101", "SEASON_STR": "2023-24", "TEAM_ABBR": "BOS",
         "OPP_ABBR": "TOR", "HOME_FLAG": False, "POINTS": 0, "X": 10, "LOCATION_VALID": True},
        {"EVENT_ID": "1021", "GAME_ID": "102", "SEASON_STR": "2023-24", "TEAM_ABBR": "TOR",
         "OPP_ABBR": "BOS", "HOME_FLAG": True, "POINTS": 3, "X": 20, "LOCATION_VALID": True},
    ])
    games = pd.DataFrame([
        {"game_id": 101, "game_date": date(2024, 1, 1), "season": 2024, "season_type": 2},
        {"game_id": 102, "game_date": date(2024, 4, 20), "season": 2024, "season_type": 3},
    ])
    return shots, games


def test_keeps_regular_attempts_without_using_observed_location_or_type():
    shots, games = source()
    original = shots.copy(deep=True)
    games = pd.concat([games, games.iloc[[0]]], ignore_index=True)
    rows, coverage = prepare_shot_forecast_rows(shots, games)
    assert list(rows.columns) == [*SHOT_COLUMNS, "GAME_DATE"]
    assert rows.EVENT_ID.tolist() == ["1011", "1012"]
    assert rows.POINTS.tolist() == [2, 0]
    assert rows.GAME_DATE.tolist() == [pd.Timestamp("2024-01-01")] * 2
    assert rows.GAME_DATE.dt.tz is None
    assert coverage == {"source_shots": 3, "source_games": 2, "metadata_games": 2,
        "unreferenced_metadata_games": 0,
        "regular_season_shots": 2, "regular_season_games": 1,
        "excluded_playoff_shots": 1, "excluded_other_phase_shots": 0}
    pd.testing.assert_frame_equal(shots, original)


def test_calendar_dates_are_eastern_even_when_source_timestamps_are_utc():
    shots, games = source()
    games["game_date"] = ["2024-01-02T02:00:00Z", "2024-04-20"]
    rows, _ = prepare_shot_forecast_rows(shots, games)
    assert rows.GAME_DATE.eq(pd.Timestamp("2024-01-01")).all()


def test_unreferenced_raw_game_phase_does_not_block_retained_shots():
    shots, games = source()
    unrelated = pd.DataFrame([{"game_id": 999, "game_date": date(2024, 4, 18),
        "season": 2024, "season_type": 5}])
    rows, coverage = prepare_shot_forecast_rows(shots, pd.concat([games, unrelated]))
    assert rows.EVENT_ID.tolist() == ["1011", "1012"]
    assert coverage["unreferenced_metadata_games"] == 1


def test_excludes_other_known_phases_and_can_return_no_regular_games():
    shots, games = source(); games.loc[0, "season_type"] = 1
    rows, coverage = prepare_shot_forecast_rows(shots, games)
    assert rows.empty and coverage["regular_season_games"] == 0
    assert coverage["excluded_other_phase_shots"] == 2 and coverage["excluded_playoff_shots"] == 1


@pytest.mark.parametrize("column,value", [
    ("EVENT_ID", None), ("GAME_ID", "unknown"), ("TEAM_ABBR", "XXX"), ("OPP_ABBR", "TOR"),
    ("HOME_FLAG", None), ("HOME_FLAG", "false"), ("HOME_FLAG", 0.5), ("POINTS", float("nan")),
    ("POINTS", float("inf")), ("POINTS", -1), ("POINTS", 1), ("POINTS", 2.5), ("POINTS", False),
    ("SEASON_STR", "2024-25"), ("SEASON_STR", None), ("SEASON_STR", "2023-2024"),
])
def test_rejects_incomplete_or_invalid_shot_rows(column, value):
    shots, games = source(); shots[column] = shots[column].astype(object); shots.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_shot_forecast_rows(shots, games)


@pytest.mark.parametrize("change", ["duplicate_event", "third_team", "conflicting_home", "missing_game"])
def test_rejects_ambiguous_event_or_game_history(change):
    shots, games = source()
    if change == "duplicate_event":
        shots.loc[1, "EVENT_ID"] = shots.loc[0, "EVENT_ID"]
    elif change == "third_team":
        shots.loc[1, "TEAM_ABBR"] = "MIA"
    elif change == "conflicting_home":
        shots.loc[1, "HOME_FLAG"] = True
    else:
        games = games.iloc[1:]
    with pytest.raises(ValueError):
        prepare_shot_forecast_rows(shots, games)


@pytest.mark.parametrize("column,value", [
    ("game_date", None), ("game_date", "tomorrow"), ("game_date", "2024-02-30"),
    ("game_date", 123), ("season", 2024.5), ("season", float("inf")), ("season", None),
    ("season_type", 9), ("season_type", True),
])
def test_rejects_invalid_game_metadata(column, value):
    shots, games = source(); games[column] = games[column].astype(object); games.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_shot_forecast_rows(shots, games)


@pytest.mark.parametrize("column,value", [("game_date", date(2024, 1, 2)), ("season", 2025), ("season_type", 3)])
def test_rejects_conflicting_metadata_for_one_game(column, value):
    shots, games = source(); duplicate = games.iloc[[0]].copy(); duplicate.loc[0, column] = value
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_shot_forecast_rows(shots, pd.concat([games, duplicate], ignore_index=True))


def test_rejects_missing_columns_and_empty_source():
    shots, games = source()
    for bad_shots, bad_games in [(shots.drop(columns="EVENT_ID"), games),
                                  (shots, games.drop(columns="game_date")), (shots.iloc[:0], games)]:
        with pytest.raises(ValueError):
            prepare_shot_forecast_rows(bad_shots, bad_games)
