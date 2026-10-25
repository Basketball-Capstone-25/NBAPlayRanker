from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from application.api_coordination.auth_dependency import require_role
from application.api_coordination.shot_data_guard import require_valid_shot_data
from domain.shot_analysis.shot_artifacts import read_shot_artifact

router = APIRouter(tags=["pbp-phase2"])


def saved_shot_result(kind: str, n_splits: int, refresh: bool = False) -> dict:
    if refresh:
        raise HTTPException(400, "Shot evaluation runs offline. Reload the saved results after the next data build.")
    require_valid_shot_data()
    try:
        result = read_shot_artifact(kind, n_splits)
    except ValueError as error:
        raise HTTPException(503, str(error) + " A new offline evaluation is required.") from error
    if kind == "analysis":
        result = {**result["payload"], **{key: result[key] for key in ("cached", "source_sha256", "n_splits", "artifact_version")}}
    result["dataset_version"] = "sha256:" + result["source_sha256"] if result.get("source_sha256") else None
    return result


@router.get("/analysis/shot-ml", dependencies=[Depends(require_role("analytics"))])
def pbp_phase2_shot_ml_analysis(
    n_splits: int = Query(5, ge=2, le=10),
    refresh: bool = Query(False),
) -> dict:
    return saved_shot_result("analysis", n_splits, refresh)


@router.get("/metrics/shot-models", dependencies=[Depends(require_role("analytics"))])
def pbp_phase2_shot_model_metrics(
    n_splits: int = Query(5, ge=2, le=10),
    refresh: bool = Query(False),
) -> dict:
    return saved_shot_result("metrics", n_splits, refresh)
