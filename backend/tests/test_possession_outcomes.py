"""Verify source weighting, overlapping rates and incomplete metric coverage."""

from types import SimpleNamespace

import pandas as pd
import pytest

from application.analytics_services.possession_outcomes import (
    build_possession_outcomes, OutcomeDataUnavailable, RATE_COLUMNS,
)
from application.api_coordination.app import rec


@pytest.mark.parametrize("season,team", [("2019-20", "PHX"), ("2024-25", "TOR"), ("2024-25", "BOS")])
def test_rates_match_raw_regular_season_player_rows(season, team):
    result = build_possession_outcomes(rec, season, team)
    raw = rec.raw_df.loc[rec.raw_df.SEASON_ID.astype(str).eq("2" + season[:4])
        & rec.raw_df.TEAM_ABBREVIATION.eq(team) & rec.raw_df.TYPE_GROUPING.eq("Offensive")]
    assert result["status"] == "ok" and result["dataset_version"] == rec.raw_df.attrs["dataset_version"]
    assert [row["play_type"] for row in result["rows"]] == sorted(raw.PLAY_TYPE.unique())
    for row in result["rows"]:
        group = raw.loc[raw.PLAY_TYPE.eq(row["play_type"])]
        assert row["possessions"] == group.POSS.sum()
        for key, column in RATE_COLUMNS.items():
            metric = row["rates"][key]
            assert metric["status"] == "ok" and metric["usable_possessions"] == row["possessions"]
            assert metric["missing_possessions"] == metric["invalid_possessions"] == 0
            assert metric["value"] == pytest.approx((group[column] * group.POSS).sum() / group.POSS.sum())


def test_real_overlap_is_not_normalized_and_small_sample_is_retained():
    phx = build_possession_outcomes(rec, "2019-20", "PHX")
    misc = next(row for row in phx["rows"] if row["play_type"] == "Misc")
    assert misc["possessions"] == 486
    assert sum(rate["value"] for rate in misc["rates"].values()) == pytest.approx(1.1870987654320986)
    bkn = build_possession_outcomes(rec, "2019-20", "BKN")
    postup = next(row for row in bkn["rows"] if row["play_type"] == "Postup")
    assert postup["possessions"] == 11 and postup["rates"]["scoring"]["value"] == pytest.approx(0.545)


def fixture_rec():
    rows = [dict(SEASON="2024-25", SEASON_ID="22024", ENTITY_TYPE="Player", PLAYER_ID=player,
        TEAM_ABBREVIATION="TOR", PLAY_TYPE="Cut", TYPE_GROUPING="Offensive", POSS=poss,
        **{column: rate for column in RATE_COLUMNS.values()})
        for player, poss, rate in [(1, 10, 0.2), (2, 90, 0.8)]]
    return SimpleNamespace(raw_df=pd.DataFrame(rows, dtype=object))


def result_row(data):
    return build_possession_outcomes(data, "2024-25", "TOR")["rows"][0]


def test_weighting_is_not_a_simple_average_and_source_is_unchanged():
    data = fixture_rec()
    before = data.raw_df.copy(deep=True)
    expected = result_row(data)
    assert expected["possessions"] == 100
    assert expected["rates"]["scoring"]["value"] == pytest.approx(0.74)
    pd.testing.assert_frame_equal(data.raw_df, before)
    data.raw_df = data.raw_df.iloc[::-1]
    assert result_row(data) == expected


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
def test_missing_positive_weight_rate_does_not_show_a_partial_average(value):
    data = fixture_rec()
    data.raw_df.loc[0, "SCORE_POSS_PCT"] = value
    row = result_row(data)
    assert row["rates"]["scoring"] == {"value": None, "status": "missing", "usable_possessions": 90,
        "missing_possessions": 10, "invalid_possessions": 0}
    assert row["rates"]["turnover"]["value"] == pytest.approx(0.74)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), -float("inf"), True, "bad"])
def test_invalid_source_rate_is_flagged_instead_of_clamped_or_averaged(value):
    data = fixture_rec()
    data.raw_df.loc[0, "SCORE_POSS_PCT"] = value
    rate = result_row(data)["rates"]["scoring"]
    assert rate == {"value": None, "status": "invalid", "usable_possessions": 90,
        "missing_possessions": 0, "invalid_possessions": 10}


def test_missing_columns_and_invalid_precedence_keep_coverage_explicit():
    data = fixture_rec()
    data.raw_df = data.raw_df.drop(columns="FT_POSS_PCT")
    data.raw_df.loc[:, "SCORE_POSS_PCT"] = [None, 1.1]
    rates = result_row(data)["rates"]
    assert rates["free_throw"] == {"value": None, "status": "missing", "usable_possessions": 0,
        "missing_possessions": 100, "invalid_possessions": 0}
    assert rates["scoring"] == {"value": None, "status": "invalid", "usable_possessions": 0,
        "missing_possessions": 10, "invalid_possessions": 90}


def test_zero_weight_rows_do_not_change_rates_and_zero_totals_stay_unavailable():
    data = fixture_rec()
    data.raw_df.loc[0, "POSS"] = 0
    data.raw_df.loc[0, "SCORE_POSS_PCT"] = 2
    assert result_row(data)["rates"]["scoring"]["value"] == 0.8
    data.raw_df.loc[1, "POSS"] = 0
    for rate in result_row(data)["rates"].values():
        assert rate == {"value": None, "status": "no_possessions", "usable_possessions": 0,
            "missing_possessions": 0, "invalid_possessions": 0}


def test_boundaries_are_valid_and_playoff_and_defensive_rows_are_excluded():
    data = fixture_rec()
    data.raw_df.loc[:, "SCORE_POSS_PCT"] = [0, 1]
    extra = data.raw_df.assign(SEASON_ID="42024", POSS=1000)
    defense = data.raw_df.assign(TYPE_GROUPING="Defensive", POSS=1000)
    data.raw_df = pd.concat([data.raw_df, extra, defense])
    row = result_row(data)
    assert row["possessions"] == 100 and row["rates"]["scoring"]["value"] == 0.9
    data.raw_df = data.raw_df.loc[data.raw_df.TYPE_GROUPING.eq("Defensive")]
    empty = build_possession_outcomes(data, "2024-25", "TOR")
    assert empty["status"] == "no_data" and empty["rows"] == []


@pytest.mark.parametrize("value", [-1, 0.1, True, None, float("nan"), float("inf"), 2**53])
def test_invalid_counts_block_the_result(value):
    data = fixture_rec()
    data.raw_df.loc[0, "POSS"] = value
    with pytest.raises(OutcomeDataUnavailable):
        result_row(data)


def test_summed_counts_must_remain_safe_for_the_browser():
    data = fixture_rec()
    data.raw_df.loc[:, "POSS"] = [2**53 - 1, 0]
    assert result_row(data)["possessions"] == 2**53 - 1
    data.raw_df.loc[1, "POSS"] = 1
    with pytest.raises(OutcomeDataUnavailable):
        result_row(data)


@pytest.mark.parametrize("change", [lambda df: df.drop(columns="POSS"),
    lambda df: pd.concat([df, df.iloc[:1]]), lambda df: df.assign(SEASON="2023-24"),
    lambda df: df.assign(SEASON_ID="bad"), lambda df: df.assign(ENTITY_TYPE="Team"),
    lambda df: df.assign(PLAYER_ID=None), lambda df: df.assign(TYPE_GROUPING="Unknown"),
    lambda df: df.assign(PLAY_TYPE=""), lambda df: df.assign(PLAY_TYPE="x" * 41),
    lambda df: pd.concat([df.assign(PLAY_TYPE=f"Play{i}") for i in range(31)])])
def test_broken_source_structure_is_unavailable(change):
    data = fixture_rec()
    data.raw_df = change(data.raw_df)
    with pytest.raises(OutcomeDataUnavailable):
        result_row(data)


@pytest.mark.parametrize("season,team", [("2024-99", "TOR"), ("bad", "TOR"),
    ("2028-29", "TOR"), ("2024-25", "XXX")])
def test_unknown_filters_are_rejected(season, team):
    with pytest.raises(ValueError):
        build_possession_outcomes(fixture_rec(), season, team)
