"""Offence/defence weight check (SCRUM-981)."""

import numpy as np
import pandas as pd
import pytest

from domain.statistical_analysis.offense_defense_weight import estimate_offense_share, team_game_table


def synthetic_shots(offense_effect: float, defense_effect: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    off = dict(zip(teams, rng.normal(0, 0.08, len(teams))))
    deff = dict(zip(teams, rng.normal(0, 0.08, len(teams))))
    rows = []
    game = 0
    for _ in range(40):
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            game += 1
            for team, opp in ((a, b), (b, a)):
                mean = 1.05 + offense_effect * off[team] + defense_effect * deff[opp]
                points = rng.normal(mean, 0.05, 80)
                rows += [{"SEASON_STR": "2024-25", "GAME_ID": game, "TEAM_ABBR": team, "OPP_ABBR": opp, "POINTS": p} for p in points]
    return pd.DataFrame(rows)


def test_leave_one_out_excludes_the_game_itself():
    shots = synthetic_shots(1.0, 1.0)
    games = team_game_table(shots)
    row = games.iloc[0]
    others = games[(games.TEAM_ABBR == row.TEAM_ABBR) & (games.GAME_ID != row.GAME_ID)]
    assert row.offense == pytest.approx(others.pts.sum() / others.n.sum())


def test_equal_effects_give_an_even_share():
    result = estimate_offense_share(synthetic_shots(1.0, 1.0), n_boot=50)
    assert result["offense_share"] == pytest.approx(0.5, abs=0.08)


def test_offense_only_effect_gives_a_large_share():
    result = estimate_offense_share(synthetic_shots(1.0, 0.0), n_boot=50)
    assert result["offense_share"] > 0.85


def test_too_few_games_are_rejected():
    with pytest.raises(ValueError):
        estimate_offense_share(synthetic_shots(1.0, 1.0).head(500))
