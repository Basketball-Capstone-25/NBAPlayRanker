"""ml_models.py

Offline ML experiment module for the Basketball Game Strategy Analysis backend.

This file defines a small pipeline that:
- builds a modeling dataset from the same Synergy CSV used by the baseline
- defines a simple statistical baseline (league-average PPP)
- fits two ML models (Ridge, RandomForest) on team play-type PPP
- evaluates all models with K-fold cross-validation (RMSE, MAE, R²)
- optionally performs a paired t-test on per-fold RMSE

You can run it directly:

    cd backend
    python3 ml_models.py

The printed table and statistics are what you reference in your defence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from baseline_recommender import BaselineRecommender

# ---------- Config ----------

# We reuse the same CSV the baseline model uses
DATA_CSV_PATH = Path(__file__).parent / "data" / "synergy_playtypes_2019_2025_players.csv"

# Features we feed into the ML models for each (season, team, play-type, offense row)
# These come from the team-level tables built in baseline_recommender.
FEATURE_COLS = [
    # Usage / volume
    "POSS",
    "POSS_PCT",
    "RELIABILITY_WEIGHT",
    # Shot quality & efficiency
    "FG_PCT",
    "EFG_PCT",
    "SCORE_POSS_PCT",
    "TOV_POSS_PCT",
    "SF_POSS_PCT",
    "FT_POSS_PCT",
    "PLUSONE_POSS_PCT",
    # League context
    "PPP_LEAGUE",
    "REL_LEAGUE",
]

# Prediction target: team’s PPP for that play type (what we ultimately care about).
TARGET_COL = "PPP"  # team's PPP for that play type


# ---------- Dataset construction ----------


def load_offense_dataset(csv_path: Path = DATA_CSV_PATH) -> pd.DataFrame:
    """Build a modeling dataset from the same preprocessed tables the baseline uses.

    Each row is a (SEASON, TEAM_ABBREVIATION, PLAY_TYPE) offense entry with:
      - Team-level Synergy stats for that play-type (PPP, FG%, usage, etc.)
      - League-average PPP and reliability for the same season + play-type.
    Our prediction target is the team's PPP for that play-type.
    """
    # Reuse existing prep logic so we stay consistent with the baseline pipeline.
    # BaselineRecommender internally builds team_df and league_df from the CSV.
    rec = BaselineRecommender(str(csv_path))
    team_df = rec.team_df
    league_df = rec.league_df

    # Offense rows only (what we actually recommend on).
    off = team_df[team_df["SIDE"] == "offense"].copy()

    # League averages for offense, keyed by (season, play_type).
    league_off = (
        league_df[league_df["SIDE"] == "offense"][
            ["SEASON", "PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]
        ]
        .rename(
            columns={
                "PPP": "PPP_LEAGUE",                 # league-average PPP for that play type
                "RELIABILITY_WEIGHT": "REL_LEAGUE",  # league reliability weight
            }
        )
    )

    # Join team offense rows with league offense context.
    data = off.merge(
        league_off,
        on=["SEASON", "PLAY_TYPE"],
        how="left",
    )

    # Drop rows with missing values in our features or target.
    # This ensures the ML models get clean numeric data.
    cols_needed = FEATURE_COLS + [TARGET_COL]
    data = data.dropna(subset=cols_needed).reset_index(drop=True)

    return data


def get_features_and_target(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Slice the modeling DataFrame into X (features) and y (target)."""
    X = data[FEATURE_COLS].to_numpy(dtype=float)
    y = data[TARGET_COL].to_numpy(dtype=float)
    return X, y


# ---------- Baseline + ML evaluation ----------


def run_cv_evaluation(
    n_splits: int = 5,
    random_state: int = 42,
    csv_path: Path = DATA_CSV_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """Run K-fold cross-validation comparing baseline vs ML models.

    Models:
      - Baseline (league mean): predicts league-average PPP per season/play-type
      - Ridge: L2-regularised linear regression
      - RandomForest: tree ensemble capturing non-linearities

    Returns:
      summary_df:
        DataFrame with one row per model and columns:
          [RMSE_mean, RMSE_std, MAE_mean, MAE_std, R2_mean, R2_std]
      fold_metrics:
        Raw per-fold metrics for each model & metric, for significance testing.
    """
    # Build the dataset from Synergy using the same pipeline as the baseline model.
    data = load_offense_dataset(csv_path)
    X, y = get_features_and_target(data)

    # K-fold cross-validation: splits data into n_splits folds.
    # Each fold becomes a test set once, and the others form the training set.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Define our models. The baseline doesn't have an estimator; it's computed directly.
    def make_ridge() -> Ridge:
        # Simple L2-regularised linear model.
        # We do not pass random_state so it stays compatible with older sklearn versions.
        return Ridge(alpha=1.0)

    def make_rf() -> RandomForestRegressor:
        # Random forest to capture non-linearities and feature interactions.
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    model_builders = {
        "Baseline (league mean)": None,  # handled separately below
        "Ridge": make_ridge,
        "RandomForest": make_rf,
    }

    # Store metrics per fold for each model.
    fold_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"RMSE": [], "MAE": [], "R2": []} for name in model_builders.keys()
    }

    # Pre-compute league-average PPP for each row for the baseline.
    # This is our simple statistical baseline: "just use the league mean."
    baseline_pred_all = data["PPP_LEAGUE"].to_numpy(dtype=float)

    # Loop over each train/test split.
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Statistical baseline: predict league-average PPP for each row ---
        y_pred_baseline = baseline_pred_all[test_idx]

        # Compute RMSE manually for compatibility with older sklearn (no 'squared' arg).
        mse_base = mean_squared_error(y_test, y_pred_baseline)
        rmse_base = float(np.sqrt(mse_base))

        fold_metrics["Baseline (league mean)"]["RMSE"].append(rmse_base)
        fold_metrics["Baseline (league mean)"]["MAE"].append(
            mean_absolute_error(y_test, y_pred_baseline)
        )
        fold_metrics["Baseline (league mean)"]["R2"].append(
            r2_score(y_test, y_pred_baseline)
        )

        # --- ML models (Ridge and RandomForest) ---
        for name, builder in model_builders.items():
            if builder is None:
                # Skip the baseline here, we already handled it above.
                continue

            # Build a fresh instance of the model for this fold.
            model = builder()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse_model = mean_squared_error(y_test, y_pred)
            rmse_model = float(np.sqrt(mse_model))

            fold_metrics[name]["RMSE"].append(rmse_model)
            fold_metrics[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
            fold_metrics[name]["R2"].append(r2_score(y_test, y_pred))

    # Build a summary table (rows = models, columns = metrics mean/std).
    rows = []
    for name, metrics in fold_metrics.items():
        row = {"model": name}
        for metric_name, values in metrics.items():
            values_arr = np.asarray(values, dtype=float)
            row[f"{metric_name}_mean"] = float(values_arr.mean())
            row[f"{metric_name}_std"] = (
                float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0
            )
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("model").sort_index()

    return summary_df, fold_metrics


# ---------- Optional: simple significance test helper ----------


def paired_t_test_rmse(
    fold_metrics: Dict[str, Dict[str, List[float]]],
    baseline_name: str = "Baseline (league mean)",
    model_name: str = "RandomForest",
) -> Tuple[float, float]:
    """Perform a paired t-test on per-fold RMSE between the baseline and a ML model.

    Returns:
      (t_statistic, p_value)

    If SciPy is not installed, falls back to computing only the t-statistic and
    sets p_value to numpy.nan.
    """
    baseline_rmse = np.asarray(fold_metrics[baseline_name]["RMSE"], dtype=float)
    model_rmse = np.asarray(fold_metrics[model_name]["RMSE"], dtype=float)
    # Differences in error for each fold: baseline - model.
    diffs = baseline_rmse - model_rmse

    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if diffs.shape[0] > 1 else 0.0
    n = diffs.shape[0]
    # Manual t-statistic in case SciPy isn't available.
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if n > 1 and std_diff > 0 else np.nan

    try:
        # If SciPy is installed, use the official implementation.
        from scipy import stats

        t_stat_scipy, p_val = stats.ttest_rel(baseline_rmse, model_rmse)
        return float(t_stat_scipy), float(p_val)
    except Exception:
        # SciPy not available: return our manual t-stat and NaN p-value.
        return float(t_stat), float("nan")


# ---------- CLI entry point ----------


if __name__ == "__main__":
    # When you run `python3 ml_models.py`, we:
    # 1. Build the dataset.
    # 2. Run cross-validated evaluation.
    # 3. Print a comparison table and a t-test.
    print("Loading data and running cross-validated comparison...")
    summary, fold_metrics = run_cv_evaluation(n_splits=5, random_state=42)

    print("\n=== Model comparison (offense PPP prediction) ===")
    print(summary)

    

    # Optional: also show a simple paired t-test between baseline and RandomForest.
    t_stat, p_val = paired_t_test_rmse(fold_metrics)
    print("\nPaired t-test on per-fold RMSE (Baseline vs RandomForest):")
    print(f"t-statistic = {t_stat:.3f}")
    if not np.isnan(p_val):
        print(f"p-value     = {p_val:.5f}")
    else:
        print("p-value     = (SciPy not installed; only t-statistic computed)")
