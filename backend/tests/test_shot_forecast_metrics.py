import numpy as np
import pandas as pd
import pytest

from domain.shot_analysis.shot_forecast_metrics import summarize_shot_errors


@pytest.fixture
def rows():
    return pd.DataFrame({
        "FGA": [4, 3], "POINTS": [5, 4],
        "ZERO_POINT_SHOTS": [2, 1], "TWO_POINT_SHOTS": [1, 2], "THREE_POINT_SHOTS": [1, 0],
    })


def test_grouped_errors_equal_each_individual_shot(rows):
    predicted = np.repeat([1.2, 0.8], [4, 3])
    observed = np.array([0, 0, 2, 3, 0, 2, 2])
    report = summarize_shot_errors(rows, [1.2, 0.8])
    assert report["shots"] == 7
    assert report["team_games"] == 2
    assert report["mae_points"] == pytest.approx(np.abs(predicted - observed).mean())
    assert report["rmse_points"] == pytest.approx(np.sqrt(((predicted - observed) ** 2).mean()))
    assert report["bias_points"] == pytest.approx((predicted - observed).mean())
    assert report["mean_observed_points"] == pytest.approx(observed.mean())
    assert report["mean_predicted_points"] == pytest.approx(predicted.mean())


@pytest.mark.parametrize("field,value", [("FGA", 0), ("FGA", 5), ("POINTS", 6),
    ("ZERO_POINT_SHOTS", -1), ("TWO_POINT_SHOTS", 1.5), ("THREE_POINT_SHOTS", np.nan)])
def test_inconsistent_counts_are_rejected(rows, field, value):
    rows = rows.astype(float)
    rows.loc[0, field] = value
    with pytest.raises(ValueError):
        summarize_shot_errors(rows, [1.2, 0.8])


@pytest.mark.parametrize("predictions", [[1.0], [[1.0], [2.0]], [np.nan, 1], [1, np.inf]])
def test_invalid_predictions_are_rejected(rows, predictions):
    with pytest.raises(ValueError, match="one finite prediction"):
        summarize_shot_errors(rows, predictions)


def test_empty_or_missing_counts_are_rejected(rows):
    for frame in [rows.iloc[:0], rows.drop(columns="POINTS")]:
        with pytest.raises(ValueError, match="non-empty"):
            summarize_shot_errors(frame, [])
