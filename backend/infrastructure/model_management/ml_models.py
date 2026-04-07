"""Offline ML training, season-holdout evaluation, and artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "POSS",
    "POSS_PCT",
    "RELIABILITY_WEIGHT",
    "FG_PCT",
    "EFG_PCT",
    "SCORE_POSS_PCT",
    "TOV_POSS_PCT",
    "SF_POSS_PCT",
    "FT_POSS_PCT",
    "PLUSONE_POSS_PCT",
    "REL_LEAGUE",
]

TARGET_COL = "PPP"

DATA_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "synergy_playtypes_2019_2025_players.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "ml_offense_ppp_predictions.csv"

RIDGE_ALPHA = 0.1
RF_PARAMS = dict(n_estimators=250, max_depth=None, min_samples_leaf=2)
GB_PARAMS = dict(learning_rate=0.1, n_estimators=400, max_depth=3)

def load_offense_dataset(team_df: pd.DataFrame, league_df: pd.DataFrame) -> pd.DataFrame:
    """Build offense modeling dataset from pre-built team and league DataFrames."""
    off = team_df[team_df["SIDE"] == "offense"].copy()

    league_off = (
        league_df[league_df["SIDE"] == "offense"][["SEASON", "PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]]
        .rename(columns={"PPP": "PPP_LEAGUE", "RELIABILITY_WEIGHT": "REL_LEAGUE"})
        .copy()
    )

    data = off.merge(league_off, on=["SEASON", "PLAY_TYPE"], how="left")

    cols_needed = FEATURE_COLS + [TARGET_COL, "PPP_LEAGUE", "SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"]
    missing = [c for c in cols_needed if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL, "PPP_LEAGUE"]).reset_index(drop=True)
    return data

def get_features_and_target(data: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X (features) and y (target PPP) from the DataFrame."""
    X = data[feature_cols].to_numpy(dtype=float)
    y = data[TARGET_COL].to_numpy(dtype=float)
    return X, y

def make_season_holdout_splits(data: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Walk-forward season-holdout CV splits."""
    seasons = sorted(data["SEASON"].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons to run season-holdout evaluation.")

    if n_splits > len(seasons) - 1:
        raise ValueError(f"n_splits={n_splits} is too large for {len(seasons)} seasons. Use <= {len(seasons)-1}.")

    start = len(seasons) - n_splits
    folds: List[Tuple[np.ndarray, np.ndarray, str]] = []

    for j in range(start, len(seasons)):
        test_season = seasons[j]
        train_seasons = seasons[:j]  # strictly earlier seasons only

        train_idx = data.index[data["SEASON"].isin(train_seasons)].to_numpy()
        test_idx = data.index[data["SEASON"] == test_season].to_numpy()

        if train_idx.size == 0 or test_idx.size == 0:
            continue

        folds.append((train_idx, test_idx, test_season))

    if not folds:
        raise ValueError("No valid season-holdout folds could be constructed.")

    return folds

def _make_ridge() -> Pipeline:

    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=RIDGE_ALPHA))])

def _make_rf(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=RF_PARAMS["n_estimators"],
        max_depth=RF_PARAMS["max_depth"],
        min_samples_leaf=RF_PARAMS["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )

def _make_gb(random_state: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        learning_rate=GB_PARAMS["learning_rate"],
        n_estimators=GB_PARAMS["n_estimators"],
        max_depth=GB_PARAMS["max_depth"],
        random_state=random_state,
    )

def run_cv_evaluation(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """Season-holdout evaluation: Baseline vs Ridge vs RF vs GB."""
    data = load_offense_dataset(team_df, league_df)

    # Use all feature cols for holdout (same set your holdout endpoint expects)
    feature_cols = list(FEATURE_COLS)

    X, y = get_features_and_target(data, feature_cols)
    folds = make_season_holdout_splits(data, n_splits=n_splits)

    model_builders: Dict[str, Callable[[], object] | None] = {
        "Baseline (league mean)": None,
        "Ridge": _make_ridge,
        "RandomForest": lambda: _make_rf(random_state),
        "GradientBoosting": lambda: _make_gb(random_state),
    }

    fold_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"RMSE": [], "MAE": [], "R2": []} for name in model_builders.keys()
    }

    baseline_pred_all = data["PPP_LEAGUE"].to_numpy(dtype=float)

    for train_idx, test_idx, _test_season in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred_baseline = baseline_pred_all[test_idx]
        fold_metrics["Baseline (league mean)"]["RMSE"].append(float(np.sqrt(mean_squared_error(y_test, y_pred_baseline))))
        fold_metrics["Baseline (league mean)"]["MAE"].append(float(mean_absolute_error(y_test, y_pred_baseline)))
        fold_metrics["Baseline (league mean)"]["R2"].append(float(r2_score(y_test, y_pred_baseline)))
        for name, builder in model_builders.items():
            if builder is None:
                continue
            model = builder()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_metrics[name]["RMSE"].append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
            fold_metrics[name]["MAE"].append(float(mean_absolute_error(y_test, y_pred)))
            fold_metrics[name]["R2"].append(float(r2_score(y_test, y_pred)))

    # Summarize
    rows = []
    for name, metrics in fold_metrics.items():
        row = {"model": name}
        for metric_name, values in metrics.items():
            arr = np.asarray(values, dtype=float)
            row[f"{metric_name}_mean"] = float(arr.mean())
            row[f"{metric_name}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("model").sort_values("RMSE_mean", ascending=True)
    return summary_df, fold_metrics

def pick_best_holdout_model_name(summary_df: pd.DataFrame) -> str:
    """Pick best ML model by lowest holdout RMSE, excluding baseline."""
    idx = [m for m in summary_df.index.tolist() if m != "Baseline (league mean)"]
    if not idx:
        return "RandomForest"
    sub = summary_df.loc[idx]
    return str(sub.sort_values("RMSE_mean", ascending=True).index[0])

def paired_t_test_rmse(
    fold_metrics: Dict[str, Dict[str, List[float]]],
    baseline_name: str = "Baseline (league mean)",
    model_name: str = "RandomForest",
) -> Tuple[float | None, float | None]:
    """Paired t-test comparing fold-level RMSE between two models."""
    base = np.asarray(fold_metrics[baseline_name]["RMSE"], dtype=float)
    mod = np.asarray(fold_metrics[model_name]["RMSE"], dtype=float)

    if base.size < 2 or mod.size < 2:
        return None, None

    diffs = base - mod
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    t_manual = mean_diff / (std_diff / np.sqrt(diffs.size)) if std_diff > 0 else None

    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(base, mod)
        return float(t_stat), float(p_val)
    except Exception:
        return (float(t_manual) if t_manual is not None else None), None

def train_and_save_predictions_walk_forward(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    random_state: int = 42,
    model_name: str = "auto",
    n_splits_for_auto: int = 5,
) -> str:
    """Walk-forward PPP_ML predictions; "auto" picks best holdout model."""
    data = load_offense_dataset(team_df, league_df)
    seasons = sorted(data["SEASON"].unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons to generate walk-forward predictions.")

    feature_cols = list(FEATURE_COLS)
    X, y = get_features_and_target(data, feature_cols)

    chosen = model_name
    if model_name.lower() == "auto":
        summary, _ = run_cv_evaluation(team_df, league_df, n_splits=n_splits_for_auto, random_state=random_state)
        chosen = pick_best_holdout_model_name(summary)

    def build_model(name: str):
        if name == "Ridge":
            return _make_ridge()
        if name == "GradientBoosting":
            return _make_gb(random_state)
        # default
        return _make_rf(random_state)

    y_hat = np.full(shape=y.shape, fill_value=np.nan, dtype=float)

    # First season fallback
    first = seasons[0]
    first_idx = data.index[data["SEASON"] == first].to_numpy()
    y_hat[first_idx] = data.loc[first_idx, "PPP_LEAGUE"].to_numpy(dtype=float)

    model = build_model(chosen)

    for s in seasons[1:]:
        train_idx = data.index[data["SEASON"].isin([x for x in seasons if x < s])].to_numpy()
        test_idx = data.index[data["SEASON"] == s].to_numpy()

        if train_idx.size == 0 or test_idx.size == 0:
            continue

        model = build_model(chosen)  # re-init each season
        model.fit(X[train_idx], y[train_idx])
        y_hat[test_idx] = model.predict(X[test_idx])

    nan_mask = np.isnan(y_hat)
    if nan_mask.any():
        y_hat[nan_mask] = data.loc[nan_mask, "PPP_LEAGUE"].to_numpy(dtype=float)

    out = data[["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"]].copy()
    out["PPP_ML"] = y_hat

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved walk-forward ML offense PPP predictions to: {output_path}")
    print(f"Artifact model used: {chosen}")
    return chosen
