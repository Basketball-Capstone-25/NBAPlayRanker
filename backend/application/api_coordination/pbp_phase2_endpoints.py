from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from application.api_coordination.auth_dependency import require_role

from domain.shot_analysis import run_shot_model_cv, compute_shot_ml_analysis

router = APIRouter(tags=["pbp-phase2"])

# Keep Dataset2 caches here (per your project rules)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "pbp" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_CACHE = CACHE_DIR / "pbp_phase2_shot_ml_analysis.json"
CV_CACHE = CACHE_DIR / "pbp_phase2_shot_model_cv.json"


def _sanitize(obj: Any) -> Any:
    """
    Recursively convert NaN/Inf to None so Starlette JSON serialization never errors.
    """
    if obj is None:
        return None

    # Fast-path for floats
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # numpy scalars sometimes show up
    try:
        import numpy as np  # local import to avoid hard dependency issues

        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]

    return obj


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@router.get("/analysis/shot-ml", dependencies=[Depends(require_role("analytics"))])
def pbp_phase2_shot_ml_analysis(
    n_splits: int = Query(5, ge=2, le=10, description="Metadata only (kept for consistency)."),
    refresh: bool = Query(False, description="If true, recompute (can take time)."),
) -> Dict[str, Any]:
    """
    Defense-friendly Phase 2 analysis payload for Dataset2.
    Cache-first so the endpoint is instant in demos.
    """
    if (not refresh) and ANALYSIS_CACHE.exists():
        cached = _read_json(ANALYSIS_CACHE)
        return jsonable_encoder(cached)

    # If no cache and refresh=false, do NOT compute on request (prevents endless loading)
    if (not refresh) and (not ANALYSIS_CACHE.exists()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Phase 2 analysis cache not found.\n"
                "Run:\n"
                "  python backend/data/etl/pbp_phase2_build.py\n"
                f"Expected cache: {ANALYSIS_CACHE}"
            ),
        )

    t0 = time.time()
    # This function may emit correlation warnings for constant columns; that’s OK.
    payload = compute_shot_ml_analysis(n_splits=int(n_splits), force_refresh=True)
    dt = time.time() - t0

    out = {
        "cached": False,
        "computed_at_unix": int(time.time()),
        "compute_seconds": float(dt),
        "payload": _sanitize(payload),
    }
    _write_json(ANALYSIS_CACHE, out)
    return jsonable_encoder(out)


@router.get("/metrics/shot-models", dependencies=[Depends(require_role("analytics"))])
def pbp_phase2_shot_model_metrics(
    n_splits: int = Query(5, ge=2, le=10, description="GroupKFold splits by GAME_ID."),
    refresh: bool = Query(False, description="If true, recompute CV (can take time)."),
) -> Dict[str, Any]:
    """
    CV evaluation summary tables for Dataset2 shot models.
    Cache-first so the endpoint is instant in demos.
    """
    if (not refresh) and CV_CACHE.exists():
        cached = _read_json(CV_CACHE)
        return jsonable_encoder(cached)

    if (not refresh) and (not CV_CACHE.exists()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Phase 2 CV cache not found.\n"
                "Run:\n"
                "  python backend/data/etl/pbp_phase2_build.py\n"
                f"Expected cache: {CV_CACHE}"
            ),
        )

    t0 = time.time()
    summary_df, fold_df = run_shot_model_cv(n_splits=int(n_splits), random_state=42)
    dt = time.time() - t0

    metrics = []
    # summary_df index is model name
    for model_name, row in summary_df.iterrows():
        metrics.append(
            {
                "model": str(model_name),
                "RMSE_mean": float(row.get("RMSE_mean", float("nan"))),
                "RMSE_std": float(row.get("RMSE_std", float("nan"))),
                "MAE_mean": float(row.get("MAE_mean", float("nan"))),
                "MAE_std": float(row.get("MAE_std", float("nan"))),
                "R2_mean": float(row.get("R2_mean", float("nan"))),
                "R2_std": float(row.get("R2_std", float("nan"))),
            }
        )

    # Best model by RMSE_mean (lower is better)
    best_model: Optional[str] = None
    try:
        best_model = str(summary_df["RMSE_mean"].idxmin())
    except Exception:
        best_model = None

    out = {
        "cached": False,
        "computed_at_unix": int(time.time()),
        "compute_seconds": float(dt),
        "n_splits": int(n_splits),
        "best_model": best_model,
        "metrics": _sanitize(metrics),
        # folds can be large; keep minimal but defense-friendly
        "fold_summary": _sanitize(
            fold_df.replace([float("inf"), float("-inf")], float("nan")).to_dict(orient="records")
        ),
    }

    _write_json(CV_CACHE, out)
    return jsonable_encoder(out)
