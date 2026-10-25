"""Shot-team regression checks against event participants and rebuilt data.

The synthetic events state the shooting-team identity independently of the
game's home team and score. Dataset checks read only the retained clean file.
"""

from pathlib import Path

import pandas as pd
import pytest

from domain.shot_analysis.shot_etl import build_shots_clean


@pytest.mark.parametrize("home_score", [0, 42])
def test_shooting_team_and_opponent_follow_event_team_id(home_score):
    raw = pd.DataFrame(
        {
            "shooting_play": [True, True],
            "season": [2025, 2025],
            "game_id": ["fixture-game", "fixture-game"],
            "team_id": [1610612761, 1610612738],
            "home_team_abbreviation": ["TOR", "TOR"],
            "away_team_abbreviation": ["BOS", "BOS"],
            "home_score": [home_score, home_score],
            "shooter_id": ["home-shooter", "away-shooter"],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "period": [1, 1],
            "clock": ["11:30", "11:00"],
        }
    )

    clean = build_shots_clean(raw).set_index("SHOOTER_ID")

    assert len(clean) == 2
    actual = clean[["TEAM_ABBR", "OPP_ABBR", "HOME_FLAG"]].to_dict("index")
    assert actual == {
        "home-shooter": {"TEAM_ABBR": "TOR", "OPP_ABBR": "BOS", "HOME_FLAG": True},
        "away-shooter": {"TEAM_ABBR": "BOS", "OPP_ABBR": "TOR", "HOME_FLAG": False},
    }


@pytest.fixture(scope="module")
def clean_shot_identities():
    source = Path(__file__).resolve().parents[1] / "data/pbp/shots_clean.parquet"
    if not source.exists():
        pytest.skip("Restore the pinned shot dataset before running dataset checks.")
    return pd.read_parquet(source, columns=["SEASON_STR", "GAME_ID", "TEAM_ABBR", "OPP_ABBR"])


@pytest.mark.integration
def test_clean_shots_never_list_the_shooting_team_as_its_opponent(clean_shot_identities):
    invalid = clean_shot_identities["TEAM_ABBR"].eq(clean_shot_identities["OPP_ABBR"])
    assert not invalid.any(), f"{int(invalid.sum())} of {len(invalid)} shots have identical team and opponent"


@pytest.mark.integration
def test_complete_game_groups_include_both_shooting_teams(clean_shot_identities):
    # A complete NBA game with at least 100 recorded shots must contain both
    # shooting teams. Tiny/partial samples are deliberately excluded.
    groups = clean_shot_identities.groupby(["SEASON_STR", "GAME_ID"])
    rows = groups.size()
    teams = groups["TEAM_ABBR"].nunique()
    complete = teams[rows >= 100]
    assert len(complete) > 0
    invalid = complete.ne(2)
    assert not invalid.any(), f"{int(invalid.sum())} of {len(complete)} substantial game groups lack two shooting teams"
