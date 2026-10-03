"""Retrospective PPP calibration using strictly later-season test rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from infrastructure.model_management.ml_models import (
    FEATURE_COLS,
    RIDGE_ALPHA,
    make_season_holdout_splits,
)


def summarize_calibration(predicted: np.ndarray, realized: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Group numeric PPP predictions, without fitting any correction on test data."""
    predicted = np.asarray(predicted, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if predicted.ndim != 1 or realized.shape != predicted.shape or not predicted.size:
        raise ValueError("Calibration requires equal, non-empty one-dimensional prediction and outcome arrays.")
    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
        raise ValueError("Calibration predictions and realized PPP must be finite.")
    if not 2 <= n_bins <= 20:
        raise ValueError("n_bins must be between 2 and 20.")

    residual = predicted - realized
    lo, hi = float(predicted.min()), float(predicted.max())
    edges = np.linspace(lo, hi, n_bins + 1) if hi > lo else np.array([lo, hi])
    assignments = np.digitize(predicted, edges[1:-1], right=False)
    bins = []
    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(selected.sum())
        bins.append({
            "lower_ppp": float(edges[index]),
            "upper_ppp": float(edges[index + 1]),
            "count": count,
            "mean_predicted_ppp": float(predicted[selected].mean()) if count else None,
            "mean_realized_ppp": float(realized[selected].mean()) if count else None,
            "bias_ppp": float(residual[selected].mean()) if count else None,
            "sparse": count < 20,
        })
    return {
        "count": int(predicted.size),
        "mean_predicted_ppp": float(predicted.mean()),
        "mean_realized_ppp": float(realized.mean()),
        "bias_ppp": float(residual.mean()),
        "mae_ppp": float(np.abs(residual).mean()),
        "rmse_ppp": float(np.sqrt(np.mean(residual ** 2))),
        "binned_absolute_bias_ppp": float(sum(b["count"] * abs(b["bias_ppp"] or 0) for b in bins) / predicted.size),
        "bins": bins,
    }


def compute_calibration(data: pd.DataFrame, n_splits: int = 5, n_bins: int = 10) -> dict[str, Any]:
    """Fit a fixed Ridge configuration separately for each chronological fold."""
    required = ["SEASON", "PLAY_TYPE", "PPP", *FEATURE_COLS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Calibration data is missing columns: {', '.join(missing)}")
    if not 1 <= n_splits <= 10:
        raise ValueError("n_splits must be between 1 and 10.")
    if not 2 <= n_bins <= 20:
        raise ValueError("n_bins must be between 2 and 20.")
    frame = data.copy().reset_index(drop=True)
    if frame.empty or frame["SEASON"].isna().any():
        raise ValueError("Calibration requires rows with known seasons.")
    values = frame[[*FEATURE_COLS, "PPP"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Calibration features and realized PPP must be finite.")
    if (frame["POSS"] < 0).any():
        raise ValueError("Calibration possession counts must not be negative.")
    splits = make_season_holdout_splits(frame, n_splits)
    predictions, outcomes, folds = [], [], []
    possession_logs = np.log1p(frame["POSS"].to_numpy(dtype=float))
    league_logs = np.log1p(frame.groupby(["SEASON", "PLAY_TYPE"])["POSS"].transform("sum").to_numpy(dtype=float))
    for train_index, test_index, test_season in splits:
        train_seasons = sorted(frame.loc[train_index, "SEASON"].unique().tolist())
        if any(season >= test_season for season in train_seasons):
            raise ValueError("Calibration training seasons must precede the test season.")
        features = values[:, :-1].copy()
        for column, logs in (("RELIABILITY_WEIGHT", possession_logs), ("REL_LEAGUE", league_logs)):
            maximum = float(logs[train_index].max())
            features[:, FEATURE_COLS.index(column)] = np.clip(logs / maximum, 0, 1) if maximum > 0 else 0
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(features[train_index], values[train_index, -1])
        predicted = model.predict(features[test_index])
        realized = values[test_index, -1]
        predictions.append(predicted)
        outcomes.append(realized)
        summary = summarize_calibration(predicted, realized, n_bins)
        folds.append({
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_rows": int(len(train_index)),
            **{key: value for key, value in summary.items() if key != "bins"},
        })
    summary = summarize_calibration(np.concatenate(predictions), np.concatenate(outcomes), n_bins)
    warnings = [
        "This is retrospective held-out evaluation using season-level features, including same-season rate statistics. It does not establish pre-game forecasting accuracy.",
        "PPP is a continuous outcome, not a probability; positive bias means overprediction and negative bias means underprediction.",
    ]
    if any(row["sparse"] for row in summary["bins"]):
        warnings.append("Bins with fewer than 20 held-out rows are sparse; do not draw strong conclusions from them.")
    return {
        "model": "Ridge",
        "evaluation": {
            "method": "expanding-window season holdout",
            "model_parameters": {"alpha": RIDGE_ALPHA},
            "parameters_source": "existing fixed Ridge configuration; no tuning or calibration correction fitted during this evaluation",
            "scaler_fit": "training rows only, refitted for each fold",
            "reliability_features": "possession-based reliability normalizations use training-only maxima in each fold",
            "unit": "points per possession (PPP)",
            "row_unit": "team / play type / offensive season",
            "weighting": "each held-out team-play-type row has equal weight",
            "binning": "equal-width predicted PPP bins; final interval includes its upper edge",
            "requested_bins": n_bins,
            "n_splits": len(folds),
            "input_rows": len(frame),
            "features": list(FEATURE_COLS),
        },
        "summary": summary,
        "folds": folds,
        "warnings": warnings,
    }
