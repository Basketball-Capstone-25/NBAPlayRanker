from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from infrastructure.model_management import FEATURE_COLS, TARGET_COL


@dataclass
class _Cache:
    n_splits: int
    min_poss: int
    payload: Dict[str, Any]


_CACHE: Optional[_Cache] = None

ANALYSIS_CACHE_PATH = Path(__file__).resolve().parent / "data" / "cache" / "ml_analysis_cache.json"
SCHEMA_VERSION = "ml_analysis_v2"


def _read_disk_cache(n_splits: int, min_poss: int) -> Optional[Dict[str, Any]]:
    try:
        if not ANALYSIS_CACHE_PATH.exists():
            return None

        raw = json.loads(ANALYSIS_CACHE_PATH.read_text(encoding="utf-8"))

        if raw.get("schema_version") != SCHEMA_VERSION:
            return None
        if int(raw.get("n_splits", -1)) != int(n_splits):
            return None
        if int(raw.get("min_poss", -1)) != int(min_poss):
            return None

        payload = raw.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_disk_cache(n_splits: int, min_poss: int, payload: Dict[str, Any]) -> None:
    try:
        ANALYSIS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANALYSIS_CACHE_PATH.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "n_splits": int(n_splits),
                    "min_poss": int(min_poss),
                    "payload": payload,
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ["min", "p25", "median", "p75", "max", "mean", "std"]}
    return {
        "min": float(np.min(x)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
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


def _feature_target_corr(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    y = df[target_col].astype(float)
    for c in feature_cols:
        x = df[c].astype(float)
        v = float(x.corr(y)) if x.notna().any() and y.notna().any() else 0.0
        out.append({"feature": c, "corr": v, "abs": abs(v)})
    out.sort(key=lambda r: r["abs"], reverse=True)
    return out


def _correlation_prune(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_corr: List[Dict[str, Any]],
    threshold: float = 0.90,
) -> Dict[str, Any]:
    abs_target = {r["feature"]: float(r["abs"]) for r in target_corr}
    corr = df[feature_cols].astype(float).corr().abs().fillna(0.0)

    kept = set(feature_cols)
    dropped: List[str] = []

    while True:
        cols = [c for c in feature_cols if c in kept]
        if len(cols) < 2:
            break
        cmat = corr.loc[cols, cols].to_numpy(copy=True)
        np.fill_diagonal(cmat, 0.0)
        max_val = float(cmat.max())
        if max_val <= threshold:
            break
        idx = np.unravel_index(np.argmax(cmat),cmat.shape)
        f1, f2 = cols[idx[0]], cols[idx[1]]
        drop = f2 if abs_target.get(f1, 0.0) >= abs_target.get(f2, 0.0) else f1
        if drop in kept:
            kept.remove(drop)
            dropped.append(drop)

    return {"threshold": float(threshold), "kept": sorted(list(kept)), "dropped": dropped}


def _walk_forward_folds(seasons: List[str], n_splits: int) -> List[Tuple[List[str], str]]:
    seasons = sorted(seasons)
    start = max(1, len(seasons) - n_splits)
    folds: List[Tuple[List[str], str]] = []
    for j in range(start, len(seasons)):
        test_season = seasons[j]
        train_seasons = seasons[:j]
        if train_seasons:
            folds.append((train_seasons, test_season))
    return folds


def compute_ml_analysis(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    min_poss: int = 25,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    global _CACHE

    if _CACHE and not force_refresh and _CACHE.n_splits == n_splits and _CACHE.min_poss == min_poss:
        return _CACHE.payload

    if not force_refresh:
        disk_payload = _read_disk_cache(n_splits=n_splits, min_poss=min_poss)
        if disk_payload:
            _CACHE = _Cache(n_splits=n_splits, min_poss=min_poss, payload=disk_payload)
            return disk_payload

    off = team_df[team_df["SIDE"] == "offense"].copy()

    league_off = (
        league_df[league_df["SIDE"] == "offense"][["SEASON", "PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]]
        .rename(columns={"PPP": "PPP_LEAGUE", "RELIABILITY_WEIGHT": "REL_LEAGUE"})
        .copy()
    )

    df = off.merge(league_off, on=["SEASON", "PLAY_TYPE"], how="left")
    df = df[df["POSS"].astype(float) >= float(min_poss)].copy()

    needed = ["SEASON", TARGET_COL, "PPP_LEAGUE"] + FEATURE_COLS
    df = df.dropna(subset=needed).reset_index(drop=True)

    dataset = {
        "rows_after_filters": int(len(df)),
        "min_poss_filter": int(min_poss),
        "n_seasons": int(df["SEASON"].nunique()),
        "seasons": sorted(df["SEASON"].unique().tolist()),
        "n_teams": int(df["TEAM_ABBREVIATION"].nunique()),
        "n_play_types": int(df["PLAY_TYPE"].nunique()),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
    }

    eda = {
        "poss": _summary_stats(df["POSS"].to_numpy(dtype=float)),
        "ppp": _summary_stats(df[TARGET_COL].to_numpy(dtype=float)),
        "hist_poss": _histogram(df["POSS"].to_numpy(dtype=float), bins=20),
        "hist_ppp": _histogram(df[TARGET_COL].to_numpy(dtype=float), bins=20),
        "missing_counts": {c: int(df[c].isna().sum()) for c in needed},
    }

    corr_cols = FEATURE_COLS + [TARGET_COL, "PPP_LEAGUE"]
    correlations = _corr_matrix(df, corr_cols)
    target_corr = _feature_target_corr(df, FEATURE_COLS, TARGET_COL)
    corr_prune = _correlation_prune(df, FEATURE_COLS, target_corr, threshold=0.90)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = df["SEASON"].astype(str).to_numpy()

    n_unique_seasons = int(df["SEASON"].nunique())
    actual_splits = min(n_splits, n_unique_seasons)

    if actual_splits < 2:
        payload = {
            "dataset": dataset,
            "eda": eda,
            "correlations": correlations,
            "target_feature_corr": target_corr,
            "feature_selection": {
                "correlation_filter": corr_prune,
                "select_k_best": {"k": 0, "selected": [], "scores": []},
                "rfe": {"selected": [], "ranking": []},
            },
            "model_selection": {
                "tuning": {
                    "cv": "Not enough distinct seasons for GroupKFold tuning.",
                    "features_used": corr_prune["kept"] if corr_prune.get("kept") else FEATURE_COLS,
                    "ridge": None,
                    "random_forest": None,
                    "gradient_boosting": None,
                },
                "notes": [
                    "Statistical analysis loaded, but tuning was skipped because fewer than 2 distinct seasons were available after filtering.",
                ],
            },
        }
        _write_disk_cache(n_splits=n_splits, min_poss=min_poss, payload=payload)
        _CACHE = _Cache(n_splits=n_splits, min_poss=min_poss, payload=payload)
        return payload

    gkf = GroupKFold(n_splits=actual_splits)

    skb = SelectKBest(score_func=f_regression, k=min(6, X.shape[1]))
    skb.fit(X, y)
    skb_scores = [{"feature": FEATURE_COLS[i], "score": float(skb.scores_[i])} for i in range(len(FEATURE_COLS))]
    skb_scores.sort(key=lambda r: r["score"], reverse=True)
    skb_selected = [FEATURE_COLS[i] for i, keep in enumerate(skb.get_support()) if keep]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rfe = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=min(6, len(FEATURE_COLS)))
    rfe.fit(Xs, y)
    rfe_selected = [FEATURE_COLS[i] for i, keep in enumerate(rfe.get_support()) if keep]
    rfe_ranking = [{"feature": FEATURE_COLS[i], "rank": int(rfe.ranking_[i])} for i in range(len(FEATURE_COLS))]

    feature_selection = {
        "correlation_filter": corr_prune,
        "select_k_best": {"k": int(skb.k), "selected": skb_selected, "scores": skb_scores},
        "rfe": {"selected": rfe_selected, "ranking": sorted(rfe_ranking, key=lambda r: r["rank"])},
    }

    model_features = corr_prune["kept"] if corr_prune.get("kept") else FEATURE_COLS
    X_m = df[model_features].to_numpy(dtype=float)

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    ridge_search = GridSearchCV(
        ridge_pipe,
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 50.0]},
        scoring="neg_root_mean_squared_error",
        cv=gkf,
        n_jobs=1,
        pre_dispatch=1,
    )
    ridge_search.fit(X_m, y, groups=groups)

    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid={"n_estimators": [150], "min_samples_leaf": [2, 5], "max_depth": [None, 10]},
        scoring="neg_root_mean_squared_error",
        cv=gkf,
        n_jobs=1,
        pre_dispatch=1,
    )
    rf_search.fit(X_m, y, groups=groups)

    gbr_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid={"n_estimators": [150, 250], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        scoring="neg_root_mean_squared_error",
        cv=gkf,
        n_jobs=1,
        pre_dispatch=1,
    )
    gbr_search.fit(X_m, y, groups=groups)

    tuning = {
        "cv": "GroupKFold by season (prevents leakage)",
        "features_used": model_features,
        "ridge": {
            "best_params": {"alpha": float(ridge_search.best_params_["model__alpha"])},
            "best_rmse": float(-ridge_search.best_score_),
        },
        "random_forest": {
            "best_params": {
                "n_estimators": int(rf_search.best_params_["n_estimators"]),
                "min_samples_leaf": int(rf_search.best_params_["min_samples_leaf"]),
                "max_depth": None if rf_search.best_params_["max_depth"] is None else int(rf_search.best_params_["max_depth"]),
            },
            "best_rmse": float(-rf_search.best_score_),
        },
        "gradient_boosting": {
            "best_params": {
                "n_estimators": int(gbr_search.best_params_["n_estimators"]),
                "learning_rate": float(gbr_search.best_params_["learning_rate"]),
                "max_depth": int(gbr_search.best_params_["max_depth"]),
            },
            "best_rmse": float(-gbr_search.best_score_),
        },
    }

    payload = {
        "dataset": dataset,
        "eda": eda,
        "correlations": correlations,
        "target_feature_corr": target_corr,
        "feature_selection": feature_selection,
        "model_selection": {
            "tuning": tuning,
            "notes": [
                "Tuning RMSE is from GroupKFold cross-validation during hyperparameter search.",
                "Holdout RMSE (Model Performance page) is a stricter test: train on earlier seasons, test on later seasons.",
                "Because holdout is harder, its RMSE can be higher than the tuning RMSE (this is expected).",
                "We choose the deployed model using holdout generalization, not the tuning score.",
            ],
        },
    }

    _write_disk_cache(n_splits=n_splits, min_poss=min_poss, payload=payload)
    _CACHE = _Cache(n_splits=n_splits, min_poss=min_poss, payload=payload)
    return payload