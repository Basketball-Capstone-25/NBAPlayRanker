"""Shot-chart ML models: train, evaluate, and predict."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from domain.shot_analysis.shot_etl import CLEAN_PARQUET

# In-process cache for shots_clean.parquet

@dataclass
class _CleanCache:
    cache_id: str
    df: pd.DataFrame

_CLEAN_CACHE: Optional[_CleanCache] = None

def _clean_cache_id() -> str:
    try:
        st = CLEAN_PARQUET.stat()
        return f"{st.st_mtime_ns}|{st.st_size}"
    except Exception:
        return "unknown"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy/raw columns into the Phase 1 *clean* schema."""

    rename = {
        # season/team
        "SEASON": "SEASON_STR",
        "SEASON_STRING": "SEASON_STR",
        "TEAM_ABBREVIATION": "TEAM_ABBR",
        "TEAM_ABBREV": "TEAM_ABBR",
        "TEAM": "TEAM_ABBR",
        "OPP_ABBREVIATION": "OPP_ABBR",
        "OPP": "OPP_ABBR",
        # shot outcomes
        "SHOT_MADE_FLAG": "MADE",
        "FGM": "MADE",
        "PTS": "POINTS",
        # types/locations
        "ACTION_TYPE": "SHOT_TYPE",
        "SHOT_DISTANCE": "DIST",
        "DISTANCE": "DIST",
        "SHOT_CLOCK": "CLOCK_SEC",
        "SHOTCLOCK": "CLOCK_SEC",
        "CLOCK": "CLOCK_SEC",
        "LOC_X": "X",
        "LOC_Y": "Y",
        "ZONE_BASIC": "ZONE",
        "SHOT_ZONE_BASIC": "ZONE",
        # ids/flags
        "PLAYER_ID": "SHOOTER_ID",
        "HOME": "HOME_FLAG",
    }

    cols = {c: rename[c] for c in df.columns if c in rename and rename[c] != c}
    if cols:
        df = df.rename(columns=cols)
    return df

def get_shots_clean_df(*, force_reload: bool = False) -> pd.DataFrame:
    """Load shots_clean.parquet once per process (cached)."""
    global _CLEAN_CACHE

    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            "shots_clean.parquet not found. Run Phase 1 build:\n"
            "  python backend/data/etl/build_pbp_pipeline.py"
        )

    cid = _clean_cache_id()
    if force_reload or _CLEAN_CACHE is None or _CLEAN_CACHE.cache_id != cid:
        wanted = [
            "SEASON_STR",
            "TEAM_ABBR",
            "OPP_ABBR",
            "HOME_FLAG",
            "PERIOD",
            "CLOCK_SEC",
            "MARGIN",
            "SHOT_TYPE",
            "ZONE",
            "SHOT_VALUE",
            "DIST",
            "ANGLE",
            "X",
            "Y",
            "MADE",
            "POINTS",
            "GAME_ID",
            "SHOOTER_ID",
        ]

        from infrastructure.data_access.dataset_metadata import read_versioned_parquet
        df = read_versioned_parquet(CLEAN_PARQUET, columns=wanted)
        df = _normalize_columns(df)
        _CLEAN_CACHE = _CleanCache(cache_id=cid, df=df)

    return _CLEAN_CACHE.df

# Feature specification

def get_feature_spec(*, include_shooter: bool = False) -> Dict[str, object]:
    """Return canonical feature + target definitions for Dataset2 ML."""

    categorical = ["TEAM_ABBR", "OPP_ABBR", "HOME_FLAG", "PERIOD", "SHOT_TYPE", "ZONE"]
    numeric = ["CLOCK_SEC", "MARGIN", "DIST", "ANGLE", "X", "Y"]

    if include_shooter:
        categorical = categorical + ["SHOOTER_ID"]

    return {
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target": "POINTS",
        "group_col": "GAME_ID",
        "prediction_kind": "proposed_shot",
        "input_timing": {
            "context_at_prediction": ["SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "HOME_FLAG", "PERIOD", "CLOCK_SEC", "MARGIN"],
            "chosen_shot": ["SHOT_TYPE", "ZONE", "DIST", "ANGLE", "X", "Y"],
            "outcome_only": ["MADE", "POINTS"],
        },
        "limitations": "Shot location and type describe a proposed shot, not information known before a possession. Clock and margin must describe the prediction instant; later event values cannot be used for a pre-possession claim. SHOT_VALUE is excluded because older sources do not identify every missed attempt as two or three points.",
    }

def shot_schema_report(df: pd.DataFrame) -> dict:
    """Describe required features and unused metadata without rejecting extras."""
    normalized = _normalize_columns(df)
    spec = get_feature_spec()
    required = set(spec["categorical_features"] + spec["numeric_features"] + [spec["target"], spec["group_col"]])
    optional = {"SEASON_STR", "SHOOTER_ID", "MADE", "SHOT_VALUE", "EVENT_ID"}
    cols = set(normalized.columns)
    return {
        "mapped": sorted(cols & required),
        "optional": sorted(cols & optional),
        "ignored": sorted(cols - required - optional),
        "missing_required": sorted(required - cols),
        "renamed": {str(before): str(after) for before, after in zip(df.columns, normalized.columns) if before != after},
    }


def load_shots_for_ml(
    *,
    max_rows: Optional[int] = 200_000,
    random_state: int = 42,
    include_shooter: bool = False,
) -> pd.DataFrame:
    """Load + lightly clean a working ML dataframe."""

    from domain.shot_analysis.shot_quality import inspect_shot_rows

    df = get_shots_clean_df().copy()
    df = _normalize_columns(df)

    spec = get_feature_spec(include_shooter=include_shooter)
    required = set(spec["categorical_features"] + spec["numeric_features"] + [spec["target"], spec["group_col"]])
    missing = [c for c in required if c not in df.columns]
    if missing:
        report = shot_schema_report(df)
        raise KeyError(f"Shot schema: missing required columns {missing}; ignored metadata {report['ignored']}")

    quality = inspect_shot_rows(df)
    if not quality["valid_attribution"]:
        raise ValueError("Shot team attribution is invalid; rebuild from original source events.")

    for c in spec["numeric_features"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[spec["target"]] = pd.to_numeric(df[spec["target"]], errors="coerce")

    # Unknown context is not a tied score. Training is blocked until the
    # source provides finite values for the proposed-shot input contract.
    incomplete = {
        c: int((~np.isfinite(df[c])).sum()) for c in spec["numeric_features"]
        if not np.isfinite(df[c]).all()
    }
    if incomplete:
        raise ValueError(f"Shot features unavailable; rebuild from complete source events: {incomplete}")

    # Drop rows missing core target + key categoricals
    core = [spec["target"], "SHOT_TYPE", "ZONE", "GAME_ID"]
    keep = [c for c in core if c in df.columns]
    df = df.dropna(subset=keep).reset_index(drop=True)

    if include_shooter and "SHOOTER_ID" in df.columns:
        vc = df["SHOOTER_ID"].value_counts(dropna=True)
        top = set(vc.head(150).index.tolist())
        df["SHOOTER_ID"] = df["SHOOTER_ID"].where(df["SHOOTER_ID"].isin(top), other="OTHER")

    if max_rows is not None and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=int(random_state)).reset_index(drop=True)

    return df

# CV utilities

def _fit_and_eval(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float]:
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

def _shot_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=1.0, random_state=int(random_state))),
            ]
        ),
        "gbr": Pipeline([("model", GradientBoostingRegressor(random_state=int(random_state)))]),
        "rf": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=18,
                        random_state=int(random_state),
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }


def run_shot_model_cv(
    n_splits: int = 5,
    random_state: int = 42,
    max_rows: Optional[int] = 75_000,
    include_shooter: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate simple regressors to predict POINTS per shot."""

    df = load_shots_for_ml(max_rows=max_rows, random_state=random_state, include_shooter=include_shooter)
    # The newest season is reserved for the separate later-season test.
    newest = sorted(df.SEASON_STR.unique())[-1]
    df = df[df.SEASON_STR.ne(newest)].reset_index(drop=True)
    spec = get_feature_spec(include_shooter=include_shooter)

    feature_cols = list(spec["categorical_features"]) + list(spec["numeric_features"])
    X_raw = df[feature_cols]
    X = pd.get_dummies(X_raw, dummy_na=True)
    y = df[spec["target"]].astype(float)

    group_col = str(spec["group_col"])
    groups = df[group_col] if group_col in df.columns else None

    models = _shot_models(random_state)

    fold_rows: List[Dict[str, float]] = []

    if groups is not None:
        splitter = GroupKFold(n_splits=int(n_splits))
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
        splits = splitter.split(X, y)

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        for name, pipe in models.items():
            r = _fit_and_eval(name, pipe, X, y, train_idx, test_idx)
            r["fold"] = float(fold_i)
            fold_rows.append(r)

    fold_df = pd.DataFrame(fold_rows)

    summary = []
    for name in models.keys():
        sub = fold_df[fold_df["model"] == name]
        summary.append(
            {
                "model": name,
                "RMSE_mean": float(sub["RMSE"].mean()),
                "RMSE_std": float(sub["RMSE"].std(ddof=0)),
                "MAE_mean": float(sub["MAE"].mean()),
                "MAE_std": float(sub["MAE"].std(ddof=0)),
                "R2_mean": float(sub["R2"].mean()),
                "R2_std": float(sub["R2"].std(ddof=0)),
            }
        )

    summary_df = pd.DataFrame(summary).set_index("model").sort_values("RMSE_mean", ascending=True)
    return summary_df, fold_df


def evaluate_shot_later_season(max_rows: int = 75_000, random_state: int = 42) -> dict:
    """Hold out the newest season; predict a specified shot, not a possession."""
    df = load_shots_for_ml(max_rows=max_rows, random_state=random_state)
    spec = get_feature_spec()
    holdout = sorted(df.SEASON_STR.unique())[-1]
    train = df.SEASON_STR.ne(holdout).to_numpy()
    test = ~train
    features = list(spec["categorical_features"]) + list(spec["numeric_features"])
    # Season is not an identity to learn when predicting an unseen season.
    features = [feature for feature in features if feature != "SEASON_STR"]
    X_train = pd.get_dummies(df.loc[train, features], dummy_na=True)
    X_test = pd.get_dummies(df.loc[test, features], dummy_na=True).reindex(columns=X_train.columns, fill_value=False)
    X = pd.concat([X_train, X_test]).sort_index()
    y = df.POINTS.astype(float)
    baseline = np.full(int(test.sum()), y[train].mean())
    rows = [{"model": "training_mean", "RMSE": float(np.sqrt(mean_squared_error(y[test], baseline))),
             "MAE": float(mean_absolute_error(y[test], baseline)), "R2": float(r2_score(y[test], baseline))}]
    for name, model in _shot_models(random_state).items():
        rows.append(_fit_and_eval(name, model, X, y, np.flatnonzero(train), np.flatnonzero(test)))
    return {"train_seasons": sorted(df.loc[train, "SEASON_STR"].unique().tolist()), "test_season": holdout,
            "train_rows": int(train.sum()), "test_rows": int(test.sum()), "max_rows": max_rows, "random_state": random_state,
            "features": features, "metrics": rows, "prediction_kind": "proposed_shot",
            "limitation": "Observed location and shot type are supplied. This does not measure pre-possession forecasts."}
