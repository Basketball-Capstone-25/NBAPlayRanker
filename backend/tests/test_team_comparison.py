"""Source parity, complete denominators and symmetric team comparisons."""

from types import SimpleNamespace

import pandas as pd
import pytest

from application.analytics_services.team_comparison import build_team_comparison, TeamComparisonDataUnavailable
from application.api_coordination.app import rec


@pytest.mark.parametrize("season,team,opponent", [
    ("2019-20", "LAL", "SAS"), ("2022-23", "DEN", "MIA"), ("2024-25", "TOR", "BOS"),
])
def test_profiles_match_raw_regular_season_totals(season, team, opponent):
    result = build_team_comparison(rec, season, team, opponent)
    source = rec.raw_df.loc[rec.raw_df.SEASON_ID.astype(str).eq("2" + season[:4])
                            & rec.raw_df.TYPE_GROUPING.eq("Offensive")]
    assert result["status"] == "ok"
    assert result["dataset_version"] == rec.raw_df.attrs["dataset_version"]
    for side, selected in [("team", team), ("opponent", opponent)]:
        team_rows = source.loc[source.TEAM_ABBREVIATION.eq(selected)]
        assert result[f"{side}_possessions"] == team_rows.POSS.sum()
        for row in result["rows"]:
            raw = team_rows.loc[team_rows.PLAY_TYPE.eq(row["play_type"])]
            assert row[side]["possessions"] == raw.POSS.sum()
            assert row[side]["points"] == raw.PTS.sum()
            assert row[side]["ppp"] == pytest.approx(raw.PTS.sum() / raw.POSS.sum())
            assert row[side]["usage_pct"] == pytest.approx(100 * raw.POSS.sum() / team_rows.POSS.sum())
        assert sum(row[side]["usage_pct"] for row in result["rows"]) == pytest.approx(100)


def fixture_rec():
    records = [("TOR", "Cut", 10, 12), ("TOR", "Spotup", 90, 81),
               ("BOS", "Cut", 40, 44), ("BOS", "Isolation", 160, 128)]
    rows = [dict(SEASON="2024-25", TEAM_ABBREVIATION=team, SIDE="offense",
                 PLAY_TYPE=play, POSS=poss, PTS=points) for team, play, poss, points in records]
    return SimpleNamespace(team_df=pd.DataFrame(rows, dtype=object), raw_df=pd.DataFrame())


def test_union_keeps_missing_categories_and_full_team_denominators():
    result = build_team_comparison(fixture_rec(), "2024-25", "TOR", "BOS")
    cut, isolation, spotup = result["rows"]
    assert [row["play_type"] for row in result["rows"]] == ["Cut", "Isolation", "Spotup"]
    assert result["team_possessions"] == 100 and result["opponent_possessions"] == 200
    assert cut["team"]["usage_pct"] == 10 and cut["opponent"]["usage_pct"] == 20
    assert cut["difference"] == pytest.approx({"usage_pp": -10, "ppp": 0.1})
    assert isolation["team"] is None and spotup["opponent"] is None
    assert isolation["difference"] == {"usage_pp": None, "ppp": None}
    assert spotup["team"]["usage_pct"] == 90


def test_swap_exchanges_profiles_and_reverses_differences():
    data = fixture_rec()
    first = build_team_comparison(data, "2024-25", "TOR", "BOS")
    swapped = build_team_comparison(data, "2024-25", "BOS", "TOR")
    assert first["team_possessions"] == swapped["opponent_possessions"]
    for a, b in zip(first["rows"], swapped["rows"]):
        assert a["play_type"] == b["play_type"]
        assert a["team"] == b["opponent"] and a["opponent"] == b["team"]
        for key, value in a["difference"].items():
            assert b["difference"][key] == (-value if value is not None else None)


def test_zero_counts_real_zero_rates_and_absent_team_are_distinct():
    data = fixture_rec()
    data.team_df.loc[0, ["POSS", "PTS"]] = 0
    zero = build_team_comparison(data, "2024-25", "TOR", "BOS")["rows"][0]
    assert zero["team"] == {"possessions": 0, "points": 0, "ppp": None, "usage_pct": 0}
    data.team_df.loc[1, "PTS"] = 0
    assert build_team_comparison(data, "2024-25", "TOR", "BOS")["rows"][2]["team"]["ppp"] == 0
    data.team_df.loc[:1, "POSS"] = 0
    empty = build_team_comparison(data, "2024-25", "TOR", "BOS")
    assert empty["team_possessions"] == 0 and empty["rows"][0]["team"]["usage_pct"] is None
    data.team_df.loc[:1, "SEASON"] = "2023-24"
    missing = build_team_comparison(data, "2024-25", "TOR", "BOS")
    assert missing["status"] == "no_data" and missing["team_possessions"] is None
    assert all(row["team"] is None for row in missing["rows"])
    assert missing["opponent_possessions"] == 200


@pytest.mark.parametrize("column,value", [("POSS", -1), ("POSS", 0.5), ("POSS", True),
    ("PTS", float("nan")), ("PTS", float("inf")), ("PTS", None), ("PTS", "bad"),
    ("POSS", 2**53), ("PTS", 2**53),
    ("PLAY_TYPE", ""), ("PLAY_TYPE", None), ("PLAY_TYPE", "x" * 41), ("POSS", 0)])
def test_invalid_source_is_unavailable(column, value):
    data = fixture_rec()
    data.team_df.loc[0, column] = value
    with pytest.raises(TeamComparisonDataUnavailable):
        build_team_comparison(data, "2024-25", "TOR", "BOS")


def test_total_counts_remain_safe_for_the_browser():
    data = fixture_rec()
    data.team_df.loc[0, "POSS"] = 2**53 - 1
    data.team_df.loc[1, ["POSS", "PTS"]] = 0
    assert build_team_comparison(data, "2024-25", "TOR", "BOS")["team_possessions"] == 2**53 - 1
    data.team_df.loc[1, "POSS"] = 1
    with pytest.raises(TeamComparisonDataUnavailable):
        build_team_comparison(data, "2024-25", "TOR", "BOS")


def test_missing_columns_duplicate_categories_and_excess_categories_fail():
    for change in [lambda df: df.drop(columns="PTS"), lambda df: pd.concat([df, df.iloc[:1]]),
                   lambda df: pd.concat([df, *[df.iloc[:1].assign(PLAY_TYPE=f"Extra{i}") for i in range(30)]])]:
        data = fixture_rec()
        data.team_df = change(data.team_df)
        with pytest.raises(TeamComparisonDataUnavailable):
            build_team_comparison(data, "2024-25", "TOR", "BOS")


@pytest.mark.parametrize("args", [("2024-99", "TOR", "BOS"), ("bad", "TOR", "BOS"),
    ("2028-29", "TOR", "BOS"), ("2024-25", "XXX", "BOS"), ("2024-25", "TOR", "XXX"),
    ("2024-25", "TOR", "TOR")])
def test_unknown_or_identical_selections_fail(args):
    with pytest.raises(ValueError):
        build_team_comparison(fixture_rec(), *args)


def test_order_does_not_change_results_or_source_data():
    data = fixture_rec()
    before = data.team_df.copy(deep=True)
    expected = build_team_comparison(data, "2024-25", "TOR", "BOS")
    pd.testing.assert_frame_equal(data.team_df, before)
    data.team_df = data.team_df.sample(frac=1, random_state=7)
    assert build_team_comparison(data, "2024-25", "TOR", "BOS") == expected
