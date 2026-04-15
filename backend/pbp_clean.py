from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pbp_constants import (
    ALT_SOURCE_PARQUET,
    CACHE_DIR,
    CANONICAL_META_JSON,
    CANONICAL_PARQUET,
    CANONICAL_SCHEMA_VERSION,
    CLEAN_PARQUET,
    SOURCE_PARQUET,
)
from pbp_cache import (
    build_meta,
    cache_valid,
    ensure_dir,
    fingerprint_file,
    write_json_atomic,
    write_parquet_atomic,
)


CANONICAL_COLUMNS: List[str] = [
    "season",
    "team",
    "opp",
    "game_id",
    "shooter_id",
    "home",
    "period",
    "clock_sec",
    "margin",
    "shot_type",
    "zone",
    "shot_value",
    "is_make",
    "points",
    "x",
    "y",
    "dist",
    "angle",
]


def _resolve_source_parquet() -> Path:
    if SOURCE_PARQUET.exists():
        return SOURCE_PARQUET
    if ALT_SOURCE_PARQUET.exists():
        return ALT_SOURCE_PARQUET
    raise FileNotFoundError(
        f"Raw parquet not found at {SOURCE_PARQUET} or {ALT_SOURCE_PARQUET}. "
        "Place it at backend/data/pbp/nba_pbp_2021_present.parquet"
    )


def _clean_parquet_looks_valid(path: Path) -> bool:
    """
    Fast sanity check so we can self-heal stale/broken shots_clean.parquet files.

    The broken file pattern in your repo was:
    - MADE all zeros
    - POINTS all zeros
    which silently poisoned heatmap/shot plan/ML/stat pages.
    """
    p = Path(path)
    if not p.exists():
        return False

    try:
        sample = pd.read_parquet(
            p,
            columns=["SEASON_STR", "TEAM_ABBR", "GAME_ID", "MADE", "POINTS", "X", "Y"],
        )
    except Exception:
        return False

    if sample.empty:
        return False

    required = {"SEASON_STR", "TEAM_ABBR", "GAME_ID", "MADE", "POINTS", "X", "Y"}
    if not required.issubset(sample.columns):
        return False

    if sample[["SEASON_STR", "TEAM_ABBR", "GAME_ID", "X", "Y"]].isna().any().any():
        return False

    made_sum = pd.to_numeric(sample["MADE"], errors="coerce").fillna(0).sum()
    points_sum = pd.to_numeric(sample["POINTS"], errors="coerce").fillna(0).sum()
    if made_sum <= 0 or points_sum <= 0:
        return False

    return True


def ensure_clean_parquet(*, force_rebuild: bool = False) -> Path:
    """Ensure `shots_clean.parquet` exists and is not obviously poisoned."""
    if CLEAN_PARQUET.exists() and not force_rebuild and _clean_parquet_looks_valid(CLEAN_PARQUET):
        return CLEAN_PARQUET

    from shot_etl import build_shots_dataset

    src = _resolve_source_parquet()
    ensure_dir(CLEAN_PARQUET.parent)
    build_shots_dataset(parquet_path=src, output_path=CLEAN_PARQUET)
    return CLEAN_PARQUET


def build_canonical_from_clean(clean_df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {
        "SEASON_STR": "season",
        "TEAM_ABBR": "team",
        "OPP_ABBR": "opp",
        "GAME_ID": "game_id",
        "SHOOTER_ID": "shooter_id",
        "HOME_FLAG": "home",
        "PERIOD": "period",
        "CLOCK_SEC": "clock_sec",
        "MARGIN": "margin",
        "SHOT_TYPE": "shot_type",
        "ZONE": "zone",
        "SHOT_VALUE": "shot_value",
        "MADE": "is_make",
        "POINTS": "points",
        "X": "x",
        "Y": "y",
        "DIST": "dist",
        "ANGLE": "angle",
    }

    df = clean_df.copy()
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[CANONICAL_COLUMNS].copy()

    for c in ["x", "y", "dist", "angle", "clock_sec", "margin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["is_make"] = pd.to_numeric(df["is_make"], errors="coerce").fillna(0).astype(int)
    df["shot_value"] = pd.to_numeric(df["shot_value"], errors="coerce").fillna(0).astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    df["home"] = df["home"].astype(bool) if df["home"].dtype == bool else df["home"].fillna(False).astype(bool)

    for c in ["shot_type", "zone"]:
        df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=["season", "team", "game_id", "x", "y"]).copy()

    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].astype(str)
    df["game_id"] = df["game_id"].astype(str)

    df["opp"] = df["opp"].where(df["opp"].notna(), None)
    df["opp"] = df["opp"].astype(object)
    df["opp"] = df["opp"].replace({"nan": None, "None": None, "NaN": None})

    df["shooter_id"] = df["shooter_id"].where(df["shooter_id"].notna(), None)
    df["shooter_id"] = df["shooter_id"].astype(object)
    df["shooter_id"] = df["shooter_id"].replace({"nan": None, "None": None, "NaN": None})

    df = df.reset_index(drop=True)
    return df


def ensure_canonical_parquet(*, force_rebuild: bool = False) -> Path:
    ensure_dir(CACHE_DIR)

    src = _resolve_source_parquet()
    fp = fingerprint_file(src, schema_version=CANONICAL_SCHEMA_VERSION)

    if (
        CANONICAL_PARQUET.exists()
        and CANONICAL_META_JSON.exists()
        and cache_valid(CANONICAL_META_JSON, fp)
        and not force_rebuild
        and _clean_parquet_looks_valid(CLEAN_PARQUET)
    ):
        return CANONICAL_PARQUET

    ensure_clean_parquet(force_rebuild=force_rebuild)

    clean_df = pd.read_parquet(CLEAN_PARQUET)
    canonical_df = build_canonical_from_clean(clean_df)

    write_parquet_atomic(canonical_df, CANONICAL_PARQUET)
    meta = build_meta(
        fingerprint=fp,
        extra={
            "rows": int(len(canonical_df)),
            "columns": CANONICAL_COLUMNS,
            "notes": "Derived from shot_etl CLEAN_PARQUET; snake_case canonical for /pbp/*.",
        },
    )
    write_json_atomic(CANONICAL_META_JSON, meta)

    return CANONICAL_PARQUET


def load_canonical_df(*, force_rebuild: bool = False) -> pd.DataFrame:
    ensure_canonical_parquet(force_rebuild=force_rebuild)
    return pd.read_parquet(CANONICAL_PARQUET)