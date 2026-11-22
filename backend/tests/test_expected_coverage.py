"""The expected seasons and teams come from the coverage file, not code."""

import json

import pytest

from application.api_coordination.data_quality_endpoints import (
    EXPECTED_SEASONS,
    EXPECTED_TEAMS,
    load_expected_coverage,
)


def test_bundled_coverage_lists_six_seasons_and_thirty_teams():
    assert EXPECTED_SEASONS[0] == "2019-20"
    assert len(EXPECTED_SEASONS) == len(set(EXPECTED_SEASONS))
    assert len(EXPECTED_TEAMS) == 30


def test_a_new_season_only_needs_a_coverage_file_change(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"seasons": ["2024-25", "2025-26"], "teams": ["TOR", "BOS"]}))

    seasons, teams = load_expected_coverage(str(path))

    assert seasons == ("2024-25", "2025-26")
    assert teams == ("TOR", "BOS")


@pytest.mark.parametrize("coverage", [{"seasons": [], "teams": ["TOR"]}, {"seasons": ["2024-25"]}])
def test_empty_coverage_is_rejected(tmp_path, coverage):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))

    with pytest.raises(ValueError):
        load_expected_coverage(str(path))
