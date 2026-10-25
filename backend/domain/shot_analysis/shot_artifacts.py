"""Read prepared shot results without fitting a model in an HTTP request."""
from functools import lru_cache
from infrastructure.data_access.dataset_metadata import source_version
import json
from pathlib import Path

from domain.shot_analysis.shot_etl import CLEAN_PARQUET

CACHE_DIR = CLEAN_PARQUET.parent / "cache"
ARTIFACT_VERSION = "proposed-shot-v3-training-only"


@lru_cache(maxsize=2)
def _source_hash(path: str, size: int, modified: int) -> str:
    return source_version(path).removeprefix("sha256:")


def source_hash(path: Path = CLEAN_PARQUET) -> str:
    stat = path.stat()
    return _source_hash(str(path), stat.st_size, stat.st_mtime_ns)


def read_shot_artifact(kind: str, n_splits: int, *, cache_dir: Path = CACHE_DIR, source: Path = CLEAN_PARQUET) -> dict:
    if kind not in {"analysis", "metrics"}:
        raise ValueError("Unsupported shot result type.")
    filename = "pbp_phase2_shot_ml_analysis.json" if kind == "analysis" else "pbp_phase2_shot_model_cv.json"
    try:
        result = json.loads((cache_dir / filename).read_text())
        if not isinstance(result, dict):
            raise ValueError("Saved shot results must contain an object.")
        if kind == "analysis" and not isinstance(result.get("payload"), dict):
            raise ValueError("Saved shot analysis payload is missing or invalid.")
        matches = (
            result.get("artifact_version") == ARTIFACT_VERSION
            and result.get("source_sha256") == source_hash(source)
            and result.get("n_splits") == n_splits
        )
        if not matches:
            raise ValueError("Saved shot results do not match the current data and fold selection.")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Saved shot results are not available for this selection.") from error
    # Return a fresh object. Never change the prepared file during a page visit.
    result["cached"] = True
    return result
