from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pbp_cache import build_meta, cache_valid, fingerprint_file, read_json, write_json_atomic
from pbp_constants import CACHE_DIR, CLEAN_PARQUET
from shot_ml_models import get_feature_spec, load_shots_for_ml


ANALYSIS_CACHE_PATH = CACHE_DIR / "shot_ml_analysis_cache.json"
SCHEMA_VERSION = "shot_ml_analysis_v2"


@dataclass
class _Cache:
    n_splits: int
    max_rows: int
    payload: Dict[str, Any]


_CACHE: Optional[_Cache] = None


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _summary_stats(x: np.ndarray) -> Dict[str, Optional[float]]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: None for k in ["min", "p25", "median", "p75", "max", "mean", "std"]}
    return {
        "min": _finite(np.min(x)),
        "p25": _finite(np.quantile(x, 0.25)),
        "median": _finite(np.quantile(x, 0.50)),
        "p75": _finite(np.quantile(x, 0.75)),
        "max": _finite(np.max(x)),
        "mean": _finite(np.mean(x)),
        "std": _finite(np.std(x, ddof=1)) if x.size > 1 else 0.0,
    }


def _histogram(x: np.ndarray, bins: int = 20) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(x, bins=bins)
    return {"bins": [float(e) for e in edges.tolist()], "counts": [int(c) for c in counts.tolist()]}


def _corr_matrix(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    sub = df[cols].astype(float)
    corr = sub.corr(method="pearson").fillna(0.0)
    return {"labels": cols, "matrix": [[float(v) for v in row] for row in corr.to_numpy().tolist()]}


def _rate(makes: float, attempts: float) -> Optional[float]:
    if attempts <= 0:
        return None
    return float(makes) / float(attempts)


def _pps(points: float, attempts: float) -> Optional[float]:
    if attempts <= 0:
        return None
    return float(points) / float(attempts)


def _build_correlation_filter(X_num: pd.DataFrame, threshold: float = 0.9) -> Tuple[List[str], List[str]]:
    cols = list(X_num.columns)
    if len(cols) <= 1:
        return cols, []

    corr = X_num.corr().abs().fillna(0.0)
    kept: List[str] = []
    dropped: List[str] = []

    for col in cols:
        if not kept:
            kept.append(col)
            continue
        too_close = any(float(corr.loc[col, k]) >= float(threshold) for k in kept if col in corr.index and k in corr.columns)
        if too_close:
            dropped.append(col)
        else:
            kept.append(col)

    return kept, dropped


def _build_select_k_best(X_num: pd.DataFrame, y: pd.Series, k: int) -> Dict[str, Any]:
    if X_num.empty:
        return {"k": 0, "selected": [], "scores": []}

    safe_k = max(1, min(int(k), X_num.shape[1]))
    skb = SelectKBest(score_func=f_regression, k=safe_k)
    skb.fit(X_num, y)
    scores = [
        {"feature": f, "score": _finite(skb.scores_[i])}
        for i, f in enumerate(X_num.columns.tolist())
    ]
    scores.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
    selected = [f for f, keep in zip(X_num.columns.tolist(), skb.get_support()) if keep]
    return {"k": int(safe_k), "selected": selected, "scores": scores}


def _build_rfe(X_num: pd.DataFrame, y: pd.Series, n_features: int) -> Dict[str, Any]:
    if X_num.empty:
        return {"selected": [], "ranking": []}

    safe_n = max(1, min(int(n_features), X_num.shape[1]))
    estimator = Ridge(alpha=1.0, random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=safe_n)
    rfe.fit(X_num, y)

    ranking = [
        {"feature": f, "rank": int(rank)}
        for f, rank in zip(X_num.columns.tolist(), rfe.ranking_.tolist())
    ]
    ranking.sort(key=lambda r: (r["rank"], r["feature"]))
    selected = [f for f, keep in zip(X_num.columns.tolist(), rfe.support_.tolist()) if keep]
    return {"selected": selected, "ranking": ranking}


def _iter_splits(X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series], n_splits: int, random_state: int):
    if groups is not None and groups.nunique() >= int(n_splits):
        splitter = GroupKFold(n_splits=int(n_splits))
        return splitter.split(X, y, groups=groups), f"GroupKFold(n_splits={int(n_splits)}, group=GAME_ID)"
    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    return splitter.split(X, y), f"KFold(n_splits={int(n_splits)}, shuffle=True)"


def _cv_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series], n_splits: int, random_state: int) -> float:
    splitter, _ = _iter_splits(X, y, groups, n_splits, random_state)
    rmses: List[float] = []
    for train_idx, test_idx in splitter:
        est = clone(model)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = est.predict(X.iloc[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("nan")


def _run_tuning(X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series], n_splits: int) -> Dict[str, Any]:
    random_state = 42
    _, cv_label = _iter_splits(X, y, groups, n_splits, random_state)

    ridge_grid = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 50.0}]
    rf_grid = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 120, "max_depth": 18},
        {"n_estimators": 160, "max_depth": None},
    ]
    gb_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 120, "learning_rate": 0.10, "max_depth": 2},
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
    ]

    def best_from_grid(kind: str, grid: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_params: Dict[str, Any] | None = None
        best_rmse = float("inf")

        for params in grid:
            if kind == "ridge":
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=float(params["alpha"]), random_state=random_state)),
                    ]
                )
            elif kind == "rf":
                model = Pipeline(
                    [
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=int(params["n_estimators"]),
                                max_depth=params["max_depth"],
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        )
                    ]
                )
            else:
                model = Pipeline(
                    [
                        (
                            "model",
                            GradientBoostingRegressor(
                                n_estimators=int(params["n_estimators"]),
                                learning_rate=float(params["learning_rate"]),
                                max_depth=int(params["max_depth"]),
                                random_state=random_state,
                            ),
                        )
                    ]
                )

            rmse = _cv_rmse(model, X, y, groups, n_splits, random_state)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

        return {"best_params": best_params or {}, "best_rmse": _finite(best_rmse)}

    return {
        "cv": cv_label,
        "features_used": X.columns.tolist(),
        "ridge": best_from_grid("ridge", ridge_grid),
        "random_forest": best_from_grid("rf", rf_grid),
        "gradient_boosting": best_from_grid("gb", gb_grid),
    }


def compute_shot_ml_analysis(
    *,
    n_splits: int = 5,
    max_rows: int = 120_000,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    global _CACHE

    if _CACHE and not force_refresh and _CACHE.n_splits == int(n_splits) and _CACHE.max_rows == int(max_rows):
        return _CACHE.payload

    fp = fingerprint_file(CLEAN_PARQUET, schema_version=SCHEMA_VERSION)
    if not force_refresh and cache_valid(ANALYSIS_CACHE_PATH, fp):
        cached = read_json(ANALYSIS_CACHE_PATH) or {}
        payload = cached.get("payload")
        if isinstance(payload, dict):
            _CACHE = _Cache(n_splits=int(n_splits), max_rows=int(max_rows), payload=payload)
            return payload

    df = load_shots_for_ml(max_rows=int(max_rows), random_state=42, include_shooter=False)
    spec = get_feature_spec(include_shooter=False)

    seasons = sorted(df["SEASON_STR"].astype(str).unique().tolist()) if "SEASON_STR" in df.columns else []
    teams = sorted(df["TEAM_ABBR"].astype(str).unique().tolist()) if "TEAM_ABBR" in df.columns else []
    shot_types = sorted(df["SHOT_TYPE"].astype(str).unique().tolist()) if "SHOT_TYPE" in df.columns else []
    zones = sorted(df["ZONE"].astype(str).unique().tolist()) if "ZONE" in df.columns else []

    dataset = {
        "rows": int(len(df)),
        "rows_used": int(len(df)),
        "max_rows": int(max_rows),
        "n_games": int(df["GAME_ID"].astype(str).nunique()) if "GAME_ID" in df.columns else 0,
        "n_seasons": int(len(seasons)),
        "seasons": seasons,
        "n_teams": int(len(teams)),
        "teams": teams,
        "n_shot_types": int(len(shot_types)),
        "shot_types": shot_types,
        "n_zones": int(len(zones)),
        "zones": zones,
        "feature_cols_numeric": list(spec["numeric_features"]),
        "feature_cols_categorical": list(spec["categorical_features"]),
        "target_col": str(spec["target"]),
        "group_col": str(spec["group_col"]),
    }

    attempts = float(len(df))
    makes = float(df["MADE"].astype(float).sum()) if "MADE" in df.columns else float("nan")
    points = float(df["POINTS"].astype(float).sum()) if "POINTS" in df.columns else float("nan")

    eda = {
        "overall": {
            "attempts": int(attempts),
            "makes": int(makes) if math.isfinite(makes) else None,
            "points": int(points) if math.isfinite(points) else None,
            "make_rate": _rate(makes, attempts),
            "points_per_shot": _pps(points, attempts),
        },
        "points": _summary_stats(df["POINTS"].to_numpy(dtype=float)),
        "dist": _summary_stats(df["DIST"].to_numpy(dtype=float)),
        "hist_points": _histogram(df["POINTS"].to_numpy(dtype=float), bins=8),
        "missing_counts": {c: int(df[c].isna().sum()) for c in list(spec["numeric_features"]) + ["ZONE", "SHOT_TYPE"] if c in df.columns},
    }

    by_type = (
        df.groupby("SHOT_TYPE", dropna=False)
        .agg(attempts=("POINTS", "size"), makes=("MADE", "sum"), points=("POINTS", "sum"))
        .reset_index()
    )
    by_type["make_rate"] = by_type.apply(lambda r: _rate(r["makes"], r["attempts"]), axis=1)
    by_type["pps"] = by_type.apply(lambda r: _pps(r["points"], r["attempts"]), axis=1)
    by_type = by_type.sort_values(["attempts"], ascending=False)

    by_zone = (
        df.groupby("ZONE", dropna=False)
        .agg(attempts=("POINTS", "size"), makes=("MADE", "sum"), points=("POINTS", "sum"))
        .reset_index()
    )
    by_zone["make_rate"] = by_zone.apply(lambda r: _rate(r["makes"], r["attempts"]), axis=1)
    by_zone["pps"] = by_zone.apply(lambda r: _pps(r["points"], r["attempts"]), axis=1)
    by_zone = by_zone.sort_values(["attempts"], ascending=False)

    breakdowns = {
        "by_shot_type": by_type.to_dict(orient="records"),
        "by_zone": by_zone.to_dict(orient="records"),
    }

    numeric_cols = [c for c in spec["numeric_features"] if c in df.columns]
    corr_cols = numeric_cols + ["POINTS"] if "POINTS" in df.columns else numeric_cols
    correlations = _corr_matrix(df, corr_cols) if len(corr_cols) >= 2 else {"labels": [], "matrix": []}

    target = df["POINTS"].astype(float)
    feature_target_corr = []
    for c in numeric_cols:
        x = df[c].astype(float)
        v = float(x.corr(target)) if x.notna().any() and target.notna().any() else 0.0
        feature_target_corr.append({"feature": c, "corr": float(v), "abs": float(abs(v))})
    feature_target_corr.sort(key=lambda r: r["abs"], reverse=True)

    X_num = df[numeric_cols].astype(float).copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0.0)
    y = target.fillna(0.0)
    groups = df[spec["group_col"]].astype(str) if spec["group_col"] in df.columns else None

    corr_kept, corr_dropped = _build_correlation_filter(X_num, threshold=0.90)
    X_corr = X_num[corr_kept].copy() if corr_kept else X_num.copy()

    skb = _build_select_k_best(X_corr, y, k=min(5, max(1, X_corr.shape[1])))
    rfe = _build_rfe(X_corr, y, n_features=min(5, max(1, X_corr.shape[1])))

    features_for_tuning = rfe["selected"] or skb["selected"] or corr_kept or numeric_cols
    X_tune = X_num[features_for_tuning].copy()
    tuning = _run_tuning(X_tune, y, groups, int(n_splits))

    payload = {
        "dataset": dataset,
        "eda": eda,
        "breakdowns": breakdowns,
        "correlations": correlations,
        "target_feature_corr": feature_target_corr,
        "feature_target_corr": feature_target_corr,
        "feature_selection": {
            "correlation_filter": {
                "threshold": 0.90,
                "kept": corr_kept,
                "dropped": corr_dropped,
                "removed": corr_dropped,
            },
            "select_k_best": skb,
            "rfe": rfe,
        },
        "model_selection": {"tuning": tuning},
    }

    meta = build_meta(fingerprint=fp, extra={"computed_at_unix": int(pd.Timestamp.utcnow().timestamp())})
    write_json_atomic(ANALYSIS_CACHE_PATH, {**meta, "payload": payload})

    _CACHE = _Cache(n_splits=int(n_splits), max_rows=int(max_rows), payload=payload)
    return payload