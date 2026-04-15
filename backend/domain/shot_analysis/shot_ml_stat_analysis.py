"""Shot-chart statistical analysis with model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from infrastructure.data_access import build_meta, cache_valid, fingerprint_file, read_json, write_json_atomic, CACHE_DIR
from domain.shot_analysis.shot_etl import CLEAN_PARQUET
from domain.shot_analysis.shot_ml_models import get_feature_spec, load_shots_for_ml

ANALYSIS_CACHE_PATH = CACHE_DIR / "shot_ml_analysis_cache.json"
SCHEMA_VERSION = "shot_ml_analysis_v1"

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

def compute_shot_ml_analysis(
    *,
    n_splits: int = 5,
    max_rows: int = 200_000,
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
        "rows_used": int(len(df)),
        "max_rows": int(max_rows),
        "n_seasons": int(len(seasons)),
        "seasons": seasons,
        "n_teams": int(len(teams)),
        "teams": teams,
        "n_shot_types": int(len(shot_types)),
        "shot_types": shot_types,
        "n_zones": int(len(zones)),
        "zones": zones,
        "feature_spec": {
            "categorical": list(spec["categorical_features"]),
            "numeric": list(spec["numeric_features"]),
            "target": str(spec["target"]),
            "group_col": str(spec["group_col"]),
        },
        "notes": [
            "Rows are sampled for speed (see max_rows).",
            "Model evaluation is exposed separately at /metrics/shot-models (GroupKFold by GAME_ID).",
        ],
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
        "points": {"summary": _summary_stats(df["POINTS"].to_numpy(dtype=float)), "hist": _histogram(df["POINTS"].to_numpy(dtype=float), bins=8)},
        "dist": {"summary": _summary_stats(df["DIST"].to_numpy(dtype=float)), "hist": _histogram(df["DIST"].to_numpy(dtype=float), bins=20)},
        "clock_sec": {"summary": _summary_stats(df["CLOCK_SEC"].to_numpy(dtype=float)), "hist": _histogram(df["CLOCK_SEC"].to_numpy(dtype=float), bins=20)},
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

    breakdowns = {"by_shot_type": by_type.to_dict(orient="records"), "by_zone": by_zone.to_dict(orient="records")}

    numeric_cols = [c for c in spec["numeric_features"] if c in df.columns] + ["POINTS"]
    correlations = _corr_matrix(df, numeric_cols) if len(numeric_cols) >= 2 else {"labels": [], "matrix": []}

    target = df["POINTS"].astype(float)
    ft_corr = []
    for c in [c for c in spec["numeric_features"] if c in df.columns]:
        x = df[c].astype(float)
        v = float(x.corr(target)) if x.notna().any() and target.notna().any() else 0.0
        ft_corr.append({"feature": c, "corr": float(v), "abs": float(abs(v))})
    ft_corr.sort(key=lambda r: r["abs"], reverse=True)

    fs_out: Dict[str, Any] = {"method": "SelectKBest(f_regression) on numeric features", "k": 0, "scores": []}
    num_feats = [c for c in spec["numeric_features"] if c in df.columns]
    if num_feats:
        Xn_df = df[num_feats].astype(float).copy()
        med = Xn_df.median(numeric_only=True)
        Xn_df = Xn_df.fillna(med).fillna(0.0)
        Xn = Xn_df.to_numpy()
        y = target.fillna(0.0).to_numpy()
        k = min(5, len(num_feats))
        skb = SelectKBest(score_func=f_regression, k=k)
        skb.fit(Xn, y)
        scores = [{"feature": f, "score": _finite(skb.scores_[i])} for i, f in enumerate(num_feats)]
        scores.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
        fs_out = {"method": fs_out["method"], "k": int(k), "scores": scores}

    payload = {
        "dataset": dataset,
        "eda": eda,
        "breakdowns": breakdowns,
        "correlations": correlations,
        "feature_target_corr": ft_corr,
        "feature_selection": fs_out,
    }

    meta = build_meta(fingerprint=fp, extra={"computed_at_unix": int(pd.Timestamp.utcnow().timestamp())})
    write_json_atomic(ANALYSIS_CACHE_PATH, {**meta, "payload": payload})

    _CACHE = _Cache(n_splits=int(n_splits), max_rows=int(max_rows), payload=payload)
    return payload
