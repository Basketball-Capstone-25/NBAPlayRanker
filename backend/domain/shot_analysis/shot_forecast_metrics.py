"""Score a constant game-team prediction against each recorded shot outcome."""

import numpy as np
import pandas as pd

OUTCOME_COUNTS = ["ZERO_POINT_SHOTS", "TWO_POINT_SHOTS", "THREE_POINT_SHOTS"]


def summarize_shot_errors(rows: pd.DataFrame, predictions) -> dict:
    required = ["FGA", "POINTS", *OUTCOME_COUNTS]
    if rows.empty or not set(required).issubset(rows.columns):
        raise ValueError("Shot errors need non-empty rows with attempt and outcome counts.")
    counts = rows[required].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if predicted.shape != (len(rows),) or not np.isfinite(predicted).all():
        raise ValueError("Provide one finite prediction per game-team row.")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Shot outcome counts must be finite non-negative integers.")
    attempts, points, outcomes = counts[:, 0], counts[:, 1], counts[:, 2:]
    values = np.array([0.0, 2.0, 3.0])
    if (attempts <= 0).any() or not np.array_equal(outcomes.sum(axis=1), attempts):
        raise ValueError("Outcome counts must sum to each row's positive attempt count.")
    if not np.array_equal(outcomes @ values, points):
        raise ValueError("Points must match the recorded 0-, 2- and 3-point outcomes.")
    total = attempts.sum()
    errors = predicted[:, None] - values
    return {
        "shots": int(total),
        "team_games": len(rows),
        "mean_predicted_points": float(predicted @ attempts / total),
        "mean_observed_points": float(points.sum() / total),
        "mae_points": float((outcomes * np.abs(errors)).sum() / total),
        "rmse_points": float(np.sqrt((outcomes * errors ** 2).sum() / total)),
        "bias_points": float((outcomes * errors).sum() / total),
    }
