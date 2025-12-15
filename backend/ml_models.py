"""backend/ml_models.py

ML module for the capstone.

This file is designed to address:
- "Default model used without checking if it fits"
- "Pipeline to process input and cleaning raw data is not demonstrated"
- "Not enough AI model testing done"
- "Model requires substantial training before expandability"

What this file provides:
1) A **reproducible ML pipeline**:
   - Loads the same Synergy snapshot used by the app
   - Aggregates to team-level (via BaselineRecommender)
   - Builds a supervised dataset for OFFENSE PPP prediction
   - Trains ML models using a scikit-learn Pipeline (imputation + one-hot encoding)

2) **Season holdout evaluation** (time-like validation):
   - Trains on earlier seasons, tests on a later season
   - Produces per-fold RMSE/MAE/R2 metrics

3) Exports the file used by the Context+ML endpoint:
   - backend/data/ml_offense_ppp_predictions.csv
   - Columns: SEASON, TEAM_ABBREVIATION, PLAY_TYPE, PPP_ML

Important scope note:
- This is not claiming to be a perfect NBA model.
- It is a defendable and reproducible demonstration of:
  data -> preprocessing -> training -> evaluation -> deployment in an API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline_recommender import BaselineRecommender


# ----------------------------
# Paths / constants
# ----------------------------

DATA_DIR = Path(__file__).parent / "data"
SYNERGY_CSV = DATA_DIR / "synergy_playtypes_2019_2025_players.csv"
ML_PRED_OUT = DATA_DIR / "ml_offense_ppp_predictions.csv"

RANDOM_SEED = 42


# ----------------------------
# Utility: season sorting
# ----------------------------

def _season_key(season: str) -> int:
    """Convert '2019-20' -> 2019 (sortable)."""
    try:
        return int(str(season).split("-")[0])
    except Exception:
        return 0


# ----------------------------
# Build ML dataset
# ----------------------------

def load_team_level_table() -> pd.DataFrame:
    """Load the Synergy snapshot and build the team-level table.

    We reuse BaselineRecommender’s aggregation so the ML dataset matches what the app uses.
    """
    if not SYNERGY_CSV.exists():
        raise FileNotFoundError(
            f"Missing Synergy CSV at: {SYNERGY_CSV}. "
            "Make sure backend/data/synergy_playtypes_2019_2025_players.csv exists."
        )
    rec = BaselineRecommender(str(SYNERGY_CSV))
    return rec.team_df.copy()


def build_offense_ml_dataset(min_poss: float = 10.0) -> Tuple[pd.DataFrame, pd.Series]:
    """Create (X, y) for offense PPP prediction.

    Target: PPP (team offense PPP by play type)
    Features (committee-friendly):
    - categorical: team, play type, season
    - numeric: usage/possessions + a few efficiency/discipline rates

    We filter to offense rows only and to play types with at least min_poss possessions
    to reduce extreme noise.
    """
    df = load_team_level_table()

    df = df[df["SIDE"] == "offense"].copy()
    df = df[df["POSS"].fillna(0) >= float(min_poss)].copy()

    # Keep only columns we want.
    feature_cols = [
        "SEASON",
        "TEAM_ABBREVIATION",
        "PLAY_TYPE",
        "GP",
        "POSS",
        "POSS_PCT",
        "FG_PCT",
        "EFG_PCT",
        "SCORE_POSS_PCT",
        "TOV_POSS_PCT",
        "SF_POSS_PCT",
        "FT_POSS_PCT",
        "PLUSONE_POSS_PCT",
    ]

    # Some columns may not exist in every snapshot; keep what exists.
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Target
    if "PPP" not in df.columns:
        raise ValueError("Expected 'PPP' column in team-level table.")
    y = pd.to_numeric(df["PPP"], errors="coerce")

    X = df[feature_cols].copy()

    # Drop rows with missing target
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return X, y


# ----------------------------
# Model pipelines
# ----------------------------

def build_baseline_model() -> Pipeline:
    """A very simple baseline: predict the global mean PPP from training data."""
    return Pipeline(
        steps=[
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def build_random_forest_model() -> Pipeline:
    """A stronger (still interpretable) ML model with proper preprocessing."""
    # Identify categorical vs numeric at runtime in fit() using column names.
    # We will build the transformer in run_cv_evaluation after we know X columns.
    # Here we just return a placeholder; we rebuild with correct columns later.
    raise RuntimeError("Use build_model_pipeline(X) instead.")


def build_model_pipeline(X: pd.DataFrame, model_kind: str) -> Pipeline:
    """Build a full preprocessing + model pipeline for the given model_kind."""
    # Categorical and numeric columns
    cat_cols = [c for c in ["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    if model_kind == "baseline_mean":
        model = DummyRegressor(strategy="mean")

    elif model_kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )

    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )

    return pipe


# ----------------------------
# Evaluation: season holdout CV
# ----------------------------

def run_cv_evaluation(n_splits: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run season-holdout evaluation and return:
    - summary_df: indexed by model name, with mean/std RMSE/MAE/R2
    - fold_metrics: long table (fold, test_season, model, RMSE, MAE, R2)

    Season holdout design:
    - Sort seasons chronologically
    - For each fold: train on seasons < test_season, test on that season
    - Use the last `n_splits` seasons as test folds (as available)
    """
    X, y = build_offense_ml_dataset(min_poss=10.0)

    seasons = sorted(X["SEASON"].dropna().unique().tolist(), key=_season_key)
    if len(seasons) < 3:
        raise ValueError("Not enough seasons to run seasonal holdout evaluation.")

    # Choose test seasons from the end (time-like validation).
    # We need at least 1 earlier season to train on.
    max_folds = max(1, len(seasons) - 1)
    n_splits = int(min(n_splits, max_folds))

    test_seasons = seasons[-n_splits:]

    model_kinds = ["baseline_mean", "random_forest"]
    model_names = {"baseline_mean": "BaselineMean", "random_forest": "RandomForest"}

    records: List[Dict[str, float]] = []

    for fold_idx, test_season in enumerate(test_seasons, start=1):
        train_mask = X["SEASON"].apply(_season_key) < _season_key(test_season)
        test_mask = X["SEASON"] == test_season

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 50 or len(X_test) < 10:
            # Skip folds that are too small to be meaningful.
            continue

        for kind in model_kinds:
            pipe = build_model_pipeline(X_train, kind)
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r2 = float(r2_score(y_test, pred))

            records.append(
                {
                    "fold": float(fold_idx),
                    "test_season": test_season,
                    "model": model_names[kind],
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                }
            )

    fold_metrics = pd.DataFrame.from_records(records)
    if fold_metrics.empty:
        raise ValueError("No valid folds produced metrics (dataset too small after filters).")

    # Summary (mean/std per model)
    summary = (
        fold_metrics.groupby("model")[["RMSE", "MAE", "R2"]]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean", "std": "std"})
    )

    # Flatten columns
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.rename(
        columns={
            "RMSE_mean": "RMSE_mean",
            "RMSE_std": "RMSE_std",
            "MAE_mean": "MAE_mean",
            "MAE_std": "MAE_std",
            "R2_mean": "R2_mean",
            "R2_std": "R2_std",
        }
    )

    # Ensure required columns exist for app.py
    for c in ["RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"]:
        if c not in summary.columns:
            summary[c] = np.nan

    # Index by model name already
    summary_df = summary

    return summary_df, fold_metrics


def paired_t_test_rmse(fold_metrics: pd.DataFrame) -> Tuple[float, float]:
    """Paired t-test comparing RMSE across folds: RandomForest vs BaselineMean.

    Returns (t_stat, p_value). If SciPy isn't available or not enough folds,
    returns (nan, nan).
    """
    try:
        from scipy.stats import ttest_rel  # type: ignore
    except Exception:
        return float("nan"), float("nan")

    if fold_metrics is None or fold_metrics.empty:
        return float("nan"), float("nan")

    # Pivot so each fold has both models
    pivot = fold_metrics.pivot_table(index="test_season", columns="model", values="RMSE", aggfunc="mean")

    if "RandomForest" not in pivot.columns or "BaselineMean" not in pivot.columns:
        return float("nan"), float("nan")

    rf = pivot["RandomForest"].dropna()
    bl = pivot["BaselineMean"].dropna()

    common = rf.index.intersection(bl.index)
    if len(common) < 2:
        return float("nan"), float("nan")

    t_stat, p_val = ttest_rel(rf.loc[common], bl.loc[common])
    return float(t_stat), float(p_val)


# ----------------------------
# Export predictions for the API
# ----------------------------

def train_full_and_export_predictions(output_path: Path = ML_PRED_OUT) -> Path:
    """Train the chosen model on ALL offense rows and export predictions.

    Output columns:
      SEASON, TEAM_ABBREVIATION, PLAY_TYPE, PPP_ML

    Note:
    - This is a deployment artifact for the app demo.
    - For strict leakage-free prediction you'd generate out-of-fold predictions per season,
      but for capstone scope this is sufficient and explainable (and we still provide
      season-holdout evaluation above).
    """
    X, y = build_offense_ml_dataset(min_poss=10.0)

    pipe = build_model_pipeline(X, model_kind="random_forest")
    pipe.fit(X, y)

    pred = pipe.predict(X)

    out = pd.DataFrame(
        {
            "SEASON": X["SEASON"].astype(str),
            "TEAM_ABBREVIATION": X["TEAM_ABBREVIATION"].astype(str),
            "PLAY_TYPE": X["PLAY_TYPE"].astype(str),
            "PPP_ML": pred.astype(float),
        }
    )

    # If duplicates exist, average them (should be rare).
    out = (
        out.groupby(["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"], as_index=False)["PPP_ML"]
        .mean()
        .sort_values(["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # 1) Run evaluation (prints to console for quick checking)
    summary_df, fold_df = run_cv_evaluation(n_splits=5)
    print("\n=== Model Evaluation (Season Holdout) ===")
    print(summary_df)

    t, p = paired_t_test_rmse(fold_df)
    print(f"\nPaired t-test (RMSE): t={t:.3f} p={p:.4f}")

    # 2) Export predictions used by the Context+ML recommender
    out_path = train_full_and_export_predictions()
    print(f"\nSaved ML predictions to: {out_path}")
