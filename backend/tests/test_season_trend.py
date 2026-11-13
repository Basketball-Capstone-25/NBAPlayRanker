from types import SimpleNamespace

import pandas as pd
import pytest

from application.analytics_services.season_trend import build_season_trend, TrendDataUnavailable
from application.api_coordination.app import rec


@pytest.mark.parametrize("team,play", [("TOR", "Cut"), ("BOS", "Spotup"), ("DEN", "PRBallHandler")])
def test_trend_matches_regular_season_source_counts(team, play):
    result = build_season_trend(rec, team, play)
    assert result["status"] == "ok"
    assert result["dataset_version"] == rec.raw_df.attrs["dataset_version"]
    assert [r["season"] for r in result["rows"]] == [f"{y}-{(y + 1) % 100:02d}" for y in range(2019, 2025)]
    for row in result["rows"]:
        raw = rec.raw_df
        source = raw.loc[raw.SEASON_ID.astype(str).eq("2" + row["season"][:4])
                         & raw.TYPE_GROUPING.eq("Offensive") & raw.TEAM_ABBREVIATION.eq(team)]
        selected = source.loc[source.PLAY_TYPE.eq(play)]
        assert row["possessions"] == selected.POSS.sum()
        assert row["points"] == selected.PTS.sum()
        assert row["team_possessions"] == source.POSS.sum()
        assert row["usage_pct"] == pytest.approx(100 * selected.POSS.sum() / source.POSS.sum())
        assert row["ppp"] == pytest.approx(selected.PTS.sum() / selected.POSS.sum())


def fixture_rec():
    rows = [dict(SEASON=season, TEAM_ABBREVIATION="TOR", SIDE="offense", PLAY_TYPE=play, POSS=poss, PTS=points)
            for season, play, poss, points in [("2021-22", "Cut", 10, 12), ("2021-22", "Spotup", 90, 81),
                                               ("2023-24", "Spotup", 80, 70), ("2024-25", "Cut", 0, 0)]]
    return SimpleNamespace(team_df=pd.DataFrame(rows, dtype=object), raw_df=pd.DataFrame())


def test_missing_season_missing_category_and_zero_are_distinct():
    rows = build_season_trend(fixture_rec(), "TOR", "Cut")["rows"]
    assert [r["season"] for r in rows] == ["2021-22", "2022-23", "2023-24", "2024-25"]
    assert rows[0]["ppp"] == 1.2 and rows[0]["usage_pct"] == 10
    assert all(v is None for k, v in rows[1].items() if k != "season")
    assert rows[2]["team_possessions"] == 80 and rows[2]["possessions"] is None
    assert rows[2]["usage_pct"] is None and rows[2]["ppp"] is None
    assert rows[3]["possessions"] == 0 and rows[3]["points"] == 0
    assert rows[3]["ppp"] is None and rows[3]["usage_pct"] is None


def test_missing_selection_is_no_data_without_invented_zeros():
    data = fixture_rec()
    data.team_df.loc[data.team_df.PLAY_TYPE.eq("Cut"), "TEAM_ABBREVIATION"] = "BOS"
    result = build_season_trend(data, "TOR", "Cut")
    assert result["status"] == "no_data"
    assert all(r["possessions"] is None for r in result["rows"])


def test_zero_points_with_recorded_possessions_is_a_real_zero_ppp():
    data = fixture_rec()
    data.team_df.loc[0, "PTS"] = 0
    row = build_season_trend(data, "TOR", "Cut")["rows"][0]
    assert row["possessions"] == 10 and row["points"] == 0
    assert row["ppp"] == 0 and row["usage_pct"] == 10


def test_zero_play_possessions_keep_zero_usage_when_team_total_exists():
    data = fixture_rec()
    data.team_df.loc[0, ["POSS", "PTS"]] = 0
    row = build_season_trend(data, "TOR", "Cut")["rows"][0]
    assert row["team_possessions"] == 90 and row["possessions"] == 0
    assert row["usage_pct"] == 0 and row["ppp"] is None


@pytest.mark.parametrize("column,value", [("POSS", -1), ("POSS", 0.5), ("PTS", float("nan")),
    ("POSS", None), ("PTS", "bad"), ("SEASON", "2021-99"), ("SEASON", None),
    ("SEASON", 2021), ("POSS", 0)])
def test_invalid_source_counts_are_not_drawn(column, value):
    data = fixture_rec()
    data.team_df.loc[0, column] = value
    with pytest.raises(TrendDataUnavailable):
        build_season_trend(data, "TOR", "Cut")


def test_duplicate_totals_and_unknown_filters_are_rejected():
    data = fixture_rec()
    data.team_df = pd.concat([data.team_df, data.team_df.iloc[:1]])
    with pytest.raises(TrendDataUnavailable):
        build_season_trend(data, "TOR", "Cut")
    for team, play in [("XXX", "Cut"), ("TOR", "Unknown")]:
        with pytest.raises(ValueError):
            build_season_trend(rec, team, play)
