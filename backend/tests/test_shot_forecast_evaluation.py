import numpy as np
import pandas as pd
import pytest

from domain.shot_analysis.shot_forecast_features import build_pregame_rows
from domain.shot_analysis import shot_forecast_evaluation as evaluation


@pytest.fixture
def prepared():
    shots = []
    for year in range(2021, 2026):
        for day in [1, 3, 5]:
            for team, opponent, home in [("TOR", "BOS", True), ("BOS", "TOR", False)]:
                points = [0, 0, 2, 3, 2 if (year + day + home) % 2 else 0]
                for point in points:
                    shots.append({"EVENT_ID": str(len(shots) + 1), "GAME_ID": f"{year}{day}",
                        "SEASON_STR": f"{year}-{(year + 1) % 100:02d}", "GAME_DATE": pd.Timestamp(year, 11, day),
                        "TEAM_ABBR": team, "OPP_ABBR": opponent, "HOME_FLAG": home, "POINTS": point})
    return build_pregame_rows(pd.DataFrame(shots))


def test_only_earlier_seasons_fit_the_model(prepared, monkeypatch):
    rows, coverage = prepared
    fit = evaluation.fit_pregame_model
    seen = []
    def observe(training):
        seen.append(training.SEASON_STR.unique().tolist())
        return fit(training)
    monkeypatch.setattr(evaluation, "fit_pregame_model", observe)
    report, final = evaluation.evaluate_pregame_forecasts(rows, coverage)
    assert seen == [["2021-22", "2022-23", "2023-24", "2024-25"]]
    assert final.SEASON_STR.unique().tolist() == ["2025-26"]
    assert report["holdout"]["shots"] == 30
    assert all(metric["shots"] == 30 for metric in report["holdout"]["metrics"].values())
    training = rows.loc[rows.SEASON_STR < "2025-26"]
    assert report["fit"]["training_mean_points"] == pytest.approx(training.POINTS.sum() / training.FGA.sum())


def test_held_out_outcomes_cannot_change_fitted_parameters(prepared):
    rows, coverage = prepared
    original, predictions = evaluation.evaluate_pregame_forecasts(rows, coverage)
    changed = rows.copy()
    final = changed.SEASON_STR.eq("2025-26")
    changed.loc[final, "POINTS"] = 0
    changed.loc[final, "ZERO_POINT_SHOTS"] = changed.loc[final, "FGA"]
    changed.loc[final, ["TWO_POINT_SHOTS", "THREE_POINT_SHOTS"]] = 0
    after, altered = evaluation.evaluate_pregame_forecasts(changed, coverage)
    assert after["fit"] == original["fit"]
    np.testing.assert_array_equal(predictions.RIDGE_POINTS, altered.RIDGE_POINTS)
    assert after["holdout"]["metrics"] != original["holdout"]["metrics"]


def test_unavailable_history_or_overlapping_seasons_are_rejected(prepared):
    rows, coverage = prepared
    with pytest.raises(ValueError, match="scored rows"):
        evaluation.evaluate_pregame_forecasts(rows.iloc[:0], coverage)
    with pytest.raises(ValueError, match="Training games"):
        evaluation.evaluate_pregame_forecasts(rows.loc[rows.SEASON_STR.eq("2025-26")], coverage)
    with pytest.raises(ValueError, match="strictly before"):
        evaluation.evaluate_pregame_forecasts(rows.assign(HISTORY_CUTOFF_DATE=rows.GAME_DATE), coverage)
    changed = rows.copy()
    changed.loc[changed.SEASON_STR.eq("2025-26"), "GAME_DATE"] = pd.Timestamp("2024-01-01")
    changed["HISTORY_CUTOFF_DATE"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="Training games"):
        evaluation.evaluate_pregame_forecasts(changed, coverage)
