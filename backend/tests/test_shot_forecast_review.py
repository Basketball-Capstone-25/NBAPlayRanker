"""Exercise rolling history updates through feature building and evaluation."""

import numpy as np
import pandas as pd

from domain.shot_analysis.shot_forecast_evaluation import evaluate_pregame_forecasts
from domain.shot_analysis.shot_forecast_features import FEATURE_COLUMNS, build_pregame_rows


def test_heldout_raw_outcomes_update_later_history_without_refitting():
    rng = np.random.default_rng(37)
    events = []
    for year in [2023, 2024, 2025]:
        for day in [1, 3, 5]:
            for team, opponent, home in [("TOR", "BOS", True), ("BOS", "TOR", False)]:
                for outcome in rng.choice([0, 2, 3], size=8):
                    events.append({
                        "EVENT_ID": str(len(events)), "GAME_ID": f"{year}-{day}",
                        "SEASON_STR": f"{year}-{str(year + 1)[-2:]}",
                        "GAME_DATE": pd.Timestamp(year, 11, day), "TEAM_ABBR": team,
                        "OPP_ABBR": opponent, "HOME_FLAG": home, "POINTS": int(outcome),
                    })
    shots = pd.DataFrame(events)
    original_rows, original_coverage = build_pregame_rows(shots)
    original_report, original_predictions = evaluate_pregame_forecasts(original_rows, original_coverage)

    changed = shots.copy()
    first_holdout = changed.GAME_DATE.eq(pd.Timestamp("2025-11-01"))
    changed.loc[first_holdout, "POINTS"] = 3
    changed_rows, changed_coverage = build_pregame_rows(changed)
    changed_report, changed_predictions = evaluate_pregame_forecasts(changed_rows, changed_coverage)

    assert changed_report["fit"] == original_report["fit"]
    assert changed_coverage == original_coverage
    first_game = original_predictions.GAME_DATE.eq(pd.Timestamp("2025-11-01"))
    pd.testing.assert_frame_equal(
        original_predictions.loc[first_game, FEATURE_COLUMNS],
        changed_predictions.loc[first_game, FEATURE_COLUMNS],
    )
    np.testing.assert_array_equal(
        original_predictions.loc[first_game, "RIDGE_POINTS"],
        changed_predictions.loc[first_game, "RIDGE_POINTS"],
    )
    later_games = ~first_game
    assert (changed_predictions.loc[later_games, "PRIOR_LEAGUE_PPS"]
            > original_predictions.loc[later_games, "PRIOR_LEAGUE_PPS"]).all()
    np.testing.assert_array_equal(
        original_predictions.TRAINING_MEAN_POINTS, changed_predictions.TRAINING_MEAN_POINTS,
    )
    assert changed_report["holdout"]["metrics"] != original_report["holdout"]["metrics"]
