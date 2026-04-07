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

        df = pd.read_parquet(CLEAN_PARQUET, columns=wanted)
        df = _normalize_columns(df)
        _CLEAN_CACHE = _CleanCache(cache_id=cid, df=df)

    return _CLEAN_CACHE.df

# Feature specification

def get_feature_spec(*, include_shooter: bool = False) -> Dict[str, object]:
    """Return canonical feature + target definitions for Dataset2 ML."""

    categorical = ["SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "HOME_FLAG", "PERIOD", "SHOT_TYPE", "ZONE"]
    numeric = ["CLOCK_SEC", "MARGIN", "SHOT_VALUE", "DIST", "ANGLE", "X", "Y"]

    if include_shooter:
        categorical = categorical + ["SHOOTER_ID"]

    return {
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target": "POINTS",
        "group_col": "GAME_ID",
    }

def load_shots_for_ml(
    *,
    max_rows: Optional[int] = 200_000,
    random_state: int = 42,
    include_shooter: bool = False,
) -> pd.DataFrame:
    """Load + lightly clean a working ML dataframe."""

    df = get_shots_clean_df().copy()
    df = _normalize_columns(df)

    spec = get_feature_spec(include_shooter=include_shooter)
    required = set(spec["categorical_features"] + spec["numeric_features"] + [spec["target"], spec["group_col"]])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"shots_clean.parquet is missing required columns: {missing}")

    for c in spec["numeric_features"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[spec["target"]] = pd.to_numeric(df[spec["target"]], errors="coerce")

    # Impute numeric NaNs with medians (models used in CV don't accept NaNs)
    for c in spec["numeric_features"]:
        if c in df.columns:
            med = float(df[c].median()) if df[c].notna().any() else 0.0
            df[c] = df[c].fillna(med)

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

def run_shot_model_cv(
    n_splits: int = 5,
    random_state: int = 42,
    max_rows: Optional[int] = 75_000,
    include_shooter: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate simple regressors to predict POINTS per shot."""

    df = load_shots_for_ml(max_rows=max_rows, random_state=random_state, include_shooter=include_shooter)
    spec = get_feature_spec(include_shooter=include_shooter)

    feature_cols = list(spec["categorical_features"]) + list(spec["numeric_features"])
    X_raw = df[feature_cols]
    X = pd.get_dummies(X_raw, dummy_na=True)
    y = df[spec["target"]].astype(float)

    group_col = str(spec["group_col"])
    groups = df[group_col] if group_col in df.columns else None

    # Keep model zoo small + reasonably fast
    models: Dict[str, Pipeline] = {
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
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }

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
