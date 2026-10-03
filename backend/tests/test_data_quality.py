"""Data quality check module: schema, completeness, duplicates, coverage (SCRUM-478)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from infrastructure.data_access.data_quality import (
    FAIL,
    NOT_RUN,
    PASS,
    DataQualityChecker,
    run_data_quality_checks,
)

SYNERGY_CSV = Path(__file__).resolve().parents[1] / "data" / "synergy_playtypes_2019_2025_players.csv"


def _frame(**overrides) -> pd.DataFrame:
    rows = [
        {"SEASON": "2023-24", "SEASON_ID": 22023, "PLAYER_ID": 1, "TEAM_ABBREVIATION": "TOR",
         "PLAY_TYPE": "Spotup", "TYPE_GROUPING": "Offensive", "POSS": 10, "PPP": 1.1, "PTS": 11},
        {"SEASON": "2023-24", "SEASON_ID": 22023, "PLAYER_ID": 2, "TEAM_ABBREVIATION": "BOS",
         "PLAY_TYPE": "Spotup", "TYPE_GROUPING": "Offensive", "POSS": 20, "PPP": 1.0, "PTS": 20},
    ]
    df = pd.DataFrame(rows)
    for column, value in overrides.items():
        df[column] = value
    return df


def _check(report, name):
    return next(check for check in report.checks if check.name == name)


def test_clean_data_passes_every_check():
    report = run_data_quality_checks(_frame(), seasons=["2023-24"], teams=["TOR", "BOS"])

    assert report.status == PASS
    assert report.safe_for_use is True
    assert [check.status for check in report.checks] == [PASS, PASS, PASS, PASS]


def test_missing_column_fails_schema_and_marks_dependent_checks_not_run():
    df = _frame().drop(columns=["PPP"])

    report = run_data_quality_checks(df, seasons=["2023-24"], teams=["TOR", "BOS"])

    assert _check(report, "schema").status == FAIL
    assert _check(report, "schema").details["missing_columns"] == ["PPP"]
    assert _check(report, "completeness").status == NOT_RUN
    assert report.safe_for_use is False


def test_null_in_key_field_fails_completeness():
    df = _frame()
    df.loc[0, "POSS"] = None

    result = DataQualityChecker().check_completeness(df, ["SEASON", "POSS"])

    assert result.status == FAIL
    assert result.details["null_counts"] == {"POSS": 1}


def test_duplicate_rows_fail_consistency_check():
    df = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)

    result = DataQualityChecker().check_duplicates(
        df, ["SEASON_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "PLAY_TYPE", "TYPE_GROUPING"]
    )

    assert result.status == FAIL
    assert result.details["duplicate_rows"] == 1


def test_missing_team_season_fails_coverage():
    report = run_data_quality_checks(_frame(), seasons=["2023-24", "2024-25"], teams=["TOR", "BOS"])

    coverage = _check(report, "coverage")
    assert coverage.status == FAIL
    assert coverage.details["missing_season_team_pairs"] == ["2024-25:BOS", "2024-25:TOR"]
    assert report.status == FAIL


def test_coverage_without_reference_lists_is_not_run():
    report = run_data_quality_checks(_frame())

    assert _check(report, "coverage").status == NOT_RUN
    assert report.status == NOT_RUN
    assert report.safe_for_use is False


@pytest.mark.integration
def test_loaded_synergy_dataset_passes_all_checks():
    df = pd.read_csv(SYNERGY_CSV)
    seasons = sorted(df["SEASON"].unique())
    teams = sorted(df["TEAM_ABBREVIATION"].unique())

    report = run_data_quality_checks(df, seasons=seasons, teams=teams)

    assert report.to_dict()["status"] == PASS, report.to_dict()
