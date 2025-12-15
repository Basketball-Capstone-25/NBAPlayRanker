"""Offline ML experiments for the Basketball Game Strategy backend.

This module is all about the **offline** work we did for the ML part:

- Build a clean offense dataset from the Synergy CSV using the same pipeline
  as the baseline recommender.
- Compare a simple baseline (league-average PPP) against two ML models.
- Report cross-validated RMSE, MAE, and R² for each model.
- Train a RandomForest with cross-validation and save PPP predictions
  per (season, team, play-type) so the recommender can reuse them later.
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

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# Same Synergy CSV used by the baseline model.
# Keeping a single source of truth means the baseline and ML models
# are both trained on the exact same data.
DATA_CSV_PATH = Path(__file__).parent / "data" / "synergy_playtypes_2019_2025_players.csv"

# Feature set for each (season, team, play-type, offense row).
# These columns all come from the team-level tables built in baseline_recommender.
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
    # League context (how stable this play type is at league level)
    "REL_LEAGUE",
]

# Target: team-level PPP for that play type.
TARGET_COL = "PPP"


# ------------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------------


def load_offense_dataset(csv_path: Path = DATA_CSV_PATH) -> pd.DataFrame:
    """
    Build a modeling dataset for offense rows.

    What it returns:
      - One row per (SEASON, TEAM_ABBREVIATION, PLAY_TYPE, SIDE='offense').
      - Each row has:
          * team-level Synergy stats for that play type
          * league-average PPP and reliability for the same season + play type.

    How it works:
      1) Reuse BaselineRecommender to get team_df and league_df so the data
         is treated exactly the same way as in the baseline model.
      2) Filter to offensive rows only (those are what we recommend on).
      3) Merge in league-average PPP and reliability per season/play-type.
      4) Drop rows with missing values in any feature or the target.
    """
    # Reuse the same preprocessing pipeline as the baseline model.
    rec = BaselineRecommender(str(csv_path))
    team_df = rec.team_df
    league_df = rec.league_df

    # Offense rows only (these drive our play-type recommendations).
    off = team_df[team_df["SIDE"] == "offense"].copy()

    # League averages for offense, keyed by (SEASON, PLAY_TYPE).
    league_off = (
        league_df[league_df["SIDE"] == "offense"][
            ["SEASON", "PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]
        ]
        .rename(
            columns={
                "PPP": "PPP_LEAGUE",
                "RELIABILITY_WEIGHT": "REL_LEAGUE",
            }
        )
    )

    # Join team offense rows with league offense context.
    data = off.merge(
        league_off,
        on=["SEASON", "PLAY_TYPE"],
        how="left",
    )

    # Keep only rows where all features + target are present.
    cols_needed = FEATURE_COLS + [TARGET_COL]
    data = data.dropna(subset=cols_needed).reset_index(drop=True)

    return data


def get_features_and_target(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract X (features) and y (target) from the modeling DataFrame.

    What it does:
      - Pulls the numeric feature columns in FEATURE_COLS into a NumPy array.
      - Pulls the PPP target column into a separate NumPy array.
    """
    X = data[FEATURE_COLS].to_numpy(dtype=float)
    y = data[TARGET_COL].to_numpy(dtype=float)
    return X, y


# ------------------------------------------------------------------
# Baseline + ML evaluation
# ------------------------------------------------------------------


def run_cv_evaluation(
    n_splits: int = 5,
    random_state: int = 42,
    csv_path: Path = DATA_CSV_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """
    Run K-fold cross-validation comparing baseline vs ML models.

    Models we compare:
      - "Baseline (league mean)": just league-average PPP per season/play-type.
      - "Ridge": linear regression with L2 regularisation.
      - "RandomForest": tree ensemble that can pick up non-linear patterns.

    Returns:
      summary_df:
        - one row per model with:
            RMSE_mean, RMSE_std,
            MAE_mean,  MAE_std,
            R2_mean,   R2_std
      fold_metrics:
        - raw per-fold metrics for each model:
            fold_metrics[model_name]["RMSE"] -> list of RMSEs
            fold_metrics[model_name]["MAE"]  -> list of MAEs
            fold_metrics[model_name]["R2"]   -> list of R² values

    How it works (high level):
      1) Build the offense modeling dataset.
      2) Set up a KFold splitter.
      3) For each fold:
           - Train Ridge and RandomForest on the training data.
           - Use league-average PPP as the baseline prediction.
           - Compute RMSE, MAE, and R² on the test data.
      4) Aggregate fold metrics into a summary table.
    """
    # Step 1: load data and turn it into X, y.
    data = load_offense_dataset(csv_path)
    X, y = get_features_and_target(data)

    # Step 2: set up K-fold cross-validation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Small helpers so we always build fresh model instances per fold.
    def make_ridge() -> Ridge:
        # Simple L2-regularised linear model.
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

    # Model builders; baseline has no builder because it doesn't train.
    model_builders = {
        "Baseline (league mean)": None,  # handled separately
        "Ridge": make_ridge,
        "RandomForest": make_rf,
    }

    # Store RMSE, MAE, R² per fold for each model.
    fold_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"RMSE": [], "MAE": [], "R2": []} for name in model_builders.keys()
    }

    # Pre-computed league-average PPP per row for the baseline.
    baseline_pred_all = data["PPP_LEAGUE"].to_numpy(dtype=float)

    # Step 3: loop over train/test splits.
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------- Baseline model ----------
        # For each test row, we just predict the league-average PPP for that row.
        y_pred_baseline = baseline_pred_all[test_idx]

        mse_base = mean_squared_error(y_test, y_pred_baseline)
        rmse_base = float(np.sqrt(mse_base))

        fold_metrics["Baseline (league mean)"]["RMSE"].append(rmse_base)
        fold_metrics["Baseline (league mean)"]["MAE"].append(
            mean_absolute_error(y_test, y_pred_baseline)
        )
        fold_metrics["Baseline (league mean)"]["R2"].append(
            r2_score(y_test, y_pred_baseline)
        )

        # ---------- ML models (Ridge and RandomForest) ----------
        for name, builder in model_builders.items():
            # Skip baseline here because we already handled it above.
            if builder is None:
                continue

            # Build and fit a fresh model on the training fold.
            model = builder()
            model.fit(X_train, y_train)

            # Predict on the test fold.
            y_pred = model.predict(X_test)

            mse_model = mean_squared_error(y_test, y_pred)
            rmse_model = float(np.sqrt(mse_model))

            fold_metrics[name]["RMSE"].append(rmse_model)
            fold_metrics[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
            fold_metrics[name]["R2"].append(r2_score(y_test, y_pred))

    # Step 4: build a summary table from the per-fold metrics.
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


# ------------------------------------------------------------------
# Simple significance test helper
# ------------------------------------------------------------------


def paired_t_test_rmse(
    fold_metrics: Dict[str, Dict[str, List[float]]],
    baseline_name: str = "Baseline (league mean)",
    model_name: str = "RandomForest",
) -> Tuple[float, float]:
    """
    Paired t-test on per-fold RMSE between the baseline and a chosen model.

    Why we have this:
      - Cross-validation gives us multiple RMSE values per model (one per fold).
      - We can treat those as paired samples and test whether the ML model
        is *significantly* better than the baseline.

    Returns:
      (t_statistic, p_value)

      - If SciPy is installed, we return the real t-test result.
      - If not, we fall back to a manual t-statistic and set p_value to NaN.
    """
    baseline_rmse = np.asarray(fold_metrics[baseline_name]["RMSE"], dtype=float)
    model_rmse = np.asarray(fold_metrics[model_name]["RMSE"], dtype=float)
    diffs = baseline_rmse - model_rmse  # positive diffs mean model < baseline (better)

    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if diffs.shape[0] > 1 else 0.0
    n = diffs.shape[0]
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if n > 1 and std_diff > 0 else np.nan

    try:
        from scipy import stats

        t_stat_scipy, p_val = stats.ttest_rel(baseline_rmse, model_rmse)
        return float(t_stat_scipy), float(p_val)
    except Exception:
        # SciPy not available: return our manual t-stat and a dummy p-value.
        return float(t_stat), float("nan")


# ------------------------------------------------------------------
# Training RF and saving PPP predictions for the recommender
# ------------------------------------------------------------------


def train_rf_and_save_predictions_cv(
    csv_path: Path = DATA_CSV_PATH,
    output_path: Path | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Use K-fold cross-validation to generate out-of-fold RF predictions for PPP.

    What this gives us:
      - For each offense row (season, team, play-type), we get a PPP_ML value
        predicted by a model that never saw that row during training.
      - This keeps things honest and avoids information leakage.

    Output CSV columns:
        SEASON, TEAM_ABBREVIATION, PLAY_TYPE, PPP_ML

    How it works:
      1) Build the offense dataset and split into X, y.
      2) Run KFold.
      3) For each fold, train a RandomForest on train and predict on test.
      4) Store those out-of-fold predictions in y_hat.
      5) Attach y_hat to the original rows and save to CSV.
    """
    if output_path is None:
        output_path = csv_path.parent / "ml_offense_ppp_predictions.csv"

    # Step 1: load data and build X, y.
    data = load_offense_dataset(csv_path)
    X, y = get_features_and_target(data)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Holds out-of-fold predictions in the same order as y.
    y_hat = np.zeros_like(y, dtype=float)

    # Step 2–4: fill y_hat with predictions from models that did not see those rows.
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_hat[test_idx] = rf.predict(X_test)

    # Step 5: attach predictions and write them out.
    data = data.copy()
    data["PPP_ML"] = y_hat

    out_cols = ["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE", "PPP_ML"]
    data[out_cols].to_csv(output_path, index=False)
    print(f"Saved CV-based ML offense PPP predictions to {output_path}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Run cross-validated comparison of baseline vs ML.
    print("Running cross-validated comparison (offense PPP prediction)...")
    summary, fold_metrics = run_cv_evaluation(n_splits=5, random_state=42)

    print("\n=== Model comparison (offense PPP prediction) ===")
    print(summary)

    # 2) Optional: check if RandomForest is significantly better than the baseline.
    t_stat, p_val = paired_t_test_rmse(fold_metrics)
    print("\nPaired t-test on per-fold RMSE (Baseline vs RandomForest):")
    print(f"t-statistic = {t_stat:.3f}")
    if not np.isnan(p_val):
        print(f"p-value     = {p_val:.5f}")
    else:
        print("p-value     = (SciPy not installed; only t-statistic computed)")

    # 3) Generate out-of-fold RF predictions for each offense row and save them.
    train_rf_and_save_predictions_cv()
