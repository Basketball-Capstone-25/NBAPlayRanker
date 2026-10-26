"""Identify local input snapshots without inventing their original pull date."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=32)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    with Path(path).open("rb") as source:
        return "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()


def source_version(path: Path | str) -> str:
    path = Path(path)
    stat = path.stat()
    return _digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _read(path: Path | str, kind: str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    before = path.stat()
    version = source_version(path)
    frame = pd.read_csv(path, **kwargs) if kind == "csv" else pd.read_parquet(path, **kwargs)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Dataset changed while loading: {path.name}. Reload the service.")
    frame.attrs["dataset_version"] = version
    frame.attrs["source_file"] = path.name
    return frame


def read_versioned_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return _read(path, "csv", **kwargs)


def read_versioned_parquet(path: Path | str, **kwargs) -> pd.DataFrame:
    return _read(path, "parquet", **kwargs)


DATASETS = {
    "synergy": ("synergy_playtypes_2019_2025_players.csv", "Play-type source rows", "NBA Stats / Synergy via hoopR", "SEASON"),
    "ml_predictions": ("ml_offense_ppp_predictions.csv", "Saved play-type predictions", "Project play-type model", "SEASON"),
    "shots": ("pbp/shots_clean.parquet", "Cleaned shot events", "ESPN play-by-play via hoopR / sportsdataverse; project ETL", "SEASON_STR"),
    "shots_canonical": ("pbp/cache/shots_canonical.parquet", "Shot explorer events", "Project shot ETL", "season"),
    "shot_aggregates": ("pbp/shots_agg.parquet", "Team shot aggregates", "Project shot ETL", "SEASON_STR"),
    "shot_league": ("pbp/shots_agg_league.parquet", "League shot aggregates", "Project shot ETL", "SEASON_STR"),
}


def describe_dataset(dataset_id: str, root: Path = DATA_DIR, frame: pd.DataFrame | None = None) -> dict:
    relative, title, source, season_column = DATASETS[dataset_id]
    path = root / relative
    base = {"id": dataset_id, "name": title, "file": relative, "source": source}
    if frame is None and not path.is_file():
        return {**base, "status": "unavailable", "error": "Dataset file is missing.",
                "version": None, "pulled_at": None, "seasons": [], "row_count": None, "columns": []}
    if frame is None and path.suffix == ".parquet":
        before = path.stat()
        version = source_version(path)
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        row_count = parquet.metadata.num_rows
        seasons = sorted(str(value) for value in parquet.read(columns=[season_column]).column(0).unique().to_pylist() if value is not None) if season_column in columns else []
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Dataset changed while reading metadata.")
    else:
        if frame is None:
            frame = read_versioned_csv(path)
        version = frame.attrs.get("dataset_version")
        columns = list(frame.columns)
        row_count = len(frame)
        seasons = sorted(frame[season_column].dropna().astype(str).unique().tolist()) if season_column in frame else []
    if not version:
        logger.warning("Loaded dataset has no version ID: %s", path.name)
    # A timestamp is accepted only when its sidecar names these exact bytes.
    pulled_at = None
    quality_exclusions = None
    sidecar = path.with_suffix(path.suffix + ".source.json")
    if sidecar.exists():
        try:
            recorded = json.loads(sidecar.read_text())
            if not isinstance(recorded, dict):
                raise ValueError("Source metadata must be an object.")
            if version and recorded.get("version") == version:
                value = recorded.get("pulled_at")
                if not isinstance(value, str) or len(value) > 40:
                    raise ValueError("Pull date must be an ISO date or timestamp.")
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                pulled_at = value
                quality = recorded.get("quality_exclusions")
                allowed_quality = {"score_inconsistent_games", "score_inconsistent_shots", "location_conflict_rows", "geometry_only_zone_rows", "unknown_attempt_values"}
                if isinstance(quality, dict) and all(isinstance(quality.get(name), int) and quality[name] >= 0 for name in allowed_quality):
                    quality_exclusions = {name: quality[name] for name in sorted(allowed_quality)}
        except (OSError, ValueError, TypeError):
            logger.warning("Unreadable source metadata for %s", path.name)
    warnings = [] if pulled_at else ["Original pull date was not recorded for this snapshot."]
    if dataset_id.startswith("shot"):
        warnings.append("A version ID identifies the source snapshot; it does not establish prediction accuracy.")
        if quality_exclusions:
            warnings.append(f"{quality_exclusions['score_inconsistent_games']} games were excluded because their source scores are inconsistent. {quality_exclusions['location_conflict_rows']} shot locations are excluded from charts and spatial models.")
            warnings.append(f"{quality_exclusions['geometry_only_zone_rows']} shot zones use approximate geometry because the source attempt value is unknown.")
    return {**base, "status": "available" if version else "unversioned", "version": version,
            "pulled_at": pulled_at, "seasons": seasons,
            "row_count": row_count, "columns": columns, "warnings": warnings,
            "quality_exclusions": quality_exclusions}


@lru_cache(maxsize=16)
def _describe_cached(dataset_id: str, root: str, size: int, mtime_ns: int, sidecar_state: tuple) -> dict:
    return describe_dataset(dataset_id, Path(root))


def dataset_catalog(root: Path = DATA_DIR, synergy: pd.DataFrame | None = None) -> list[dict]:
    result = []
    for dataset_id, (relative, *_rest) in DATASETS.items():
        path = root / relative
        try:
            if dataset_id == "synergy" and synergy is not None:
                result.append(describe_dataset(dataset_id, root, synergy))
            elif path.exists():
                stat = path.stat()
                sidecar = path.with_suffix(path.suffix + ".source.json")
                sidecar_stat = sidecar.stat() if sidecar.exists() else None
                sidecar_state = (sidecar_stat.st_size, sidecar_stat.st_mtime_ns) if sidecar_stat else ()
                result.append(_describe_cached(dataset_id, str(root), stat.st_size, stat.st_mtime_ns, sidecar_state))
            else:
                result.append(describe_dataset(dataset_id, root))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Dataset metadata unavailable for %s (%s)", dataset_id, type(exc).__name__)
            _, title, source, _ = DATASETS[dataset_id]
            result.append({"id": dataset_id, "name": title, "file": relative, "source": source,
                           "status": "unavailable", "error": "Dataset file could not be read.",
                           "version": None, "pulled_at": None, "seasons": [], "row_count": None, "columns": []})
    return result
