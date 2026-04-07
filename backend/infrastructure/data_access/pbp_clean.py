"""Play-by-play data cleaning pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .pbp_constants import (
    ALT_SOURCE_PARQUET,
    CACHE_DIR,
    CANONICAL_META_JSON,
    CANONICAL_PARQUET,
    CANONICAL_SCHEMA_VERSION,
    CLEAN_PARQUET,
    SOURCE_PARQUET,
)
from .pbp_cache import (
    build_meta,
    cache_valid,
    ensure_dir,
    fingerprint_file,
    write_json_atomic,
    write_parquet_atomic,
)

CANONICAL_COLUMNS: List[str] = [
    # identifiers
    "season",
    "team",
    "opp",
    "game_id",
    "shooter_id",
    # context
    "home",
    "period",
    "clock_sec",
    "margin",
    # shot attributes
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
    """Resolve the raw parquet path the same way shot_etl does."""
    if SOURCE_PARQUET.exists():
        return SOURCE_PARQUET
    if ALT_SOURCE_PARQUET.exists():
        return ALT_SOURCE_PARQUET
    # Keep the error message explicit for the user
    raise FileNotFoundError(
        f"Raw parquet not found at {SOURCE_PARQUET} or {ALT_SOURCE_PARQUET}. "
        "Place it at backend/data/pbp/nba_pbp_2021_present.parquet"
    )

def ensure_clean_parquet(*, force_rebuild: bool = False) -> Path:
    """Ensure `shots_clean.parquet` exists (uppercase schema used by existing endpoints)."""
    if CLEAN_PARQUET.exists() and not force_rebuild:
        return CLEAN_PARQUET

    raise FileNotFoundError(
        f"shots_clean.parquet not found at {CLEAN_PARQUET}.\n"
        "Run the ETL pipeline first:\n"
        "  python backend/data/etl/build_pbp_pipeline.py"
    )

def build_canonical_from_clean(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the existing uppercase clean shots table to a stable snake_case schema."""

    # Mapping from existing clean columns -> canonical columns.
    # These are produced by shot_etl.build_shots_clean.
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

    # Rename known fields and create any missing canonical columns.
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep canonical ordering.
    df = df[CANONICAL_COLUMNS].copy()

    # Core numerics (convert before dropping missing core fields)
    for c in ["x", "y", "dist", "angle", "clock_sec", "margin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["period"] = pd.to_numeric(df["period"], errors="coerce")

    # booleans / ints
    # Keep `is_make` as int (0/1) for ML convenience.
    df["is_make"] = pd.to_numeric(df["is_make"], errors="coerce").fillna(0).astype(int)
    df["shot_value"] = pd.to_numeric(df["shot_value"], errors="coerce").fillna(0).astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    df["home"] = df["home"].astype(bool) if df["home"].dtype == bool else df["home"].fillna(False).astype(bool)

    # Clean up text columns.
    for c in ["shot_type", "zone"]:
        df[c] = df[c].astype(str).str.strip()

    # Drop rows that are missing critical fields (before casting to str, so NaNs are real NaNs).
    df = df.dropna(subset=["season", "team", "game_id", "x", "y"]).copy()

    # Type normalization (best-effort; do not crash on weird values).
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].astype(str)
    df["game_id"] = df["game_id"].astype(str)

    # Optional string fields: keep None instead of "nan" / "None"
    df["opp"] = df["opp"].where(df["opp"].notna(), None)
    df["opp"] = df["opp"].astype(object)
    df["opp"] = df["opp"].replace({"nan": None, "None": None, "NaN": None})

    df["shooter_id"] = df["shooter_id"].where(df["shooter_id"].notna(), None)
    df["shooter_id"] = df["shooter_id"].astype(object)
    df["shooter_id"] = df["shooter_id"].replace({"nan": None, "None": None, "NaN": None})

    df = df.reset_index(drop=True)

    return df

def ensure_canonical_parquet(*, force_rebuild: bool = False) -> Path:
    """Ensure the canonical (snake_case) dataset exists and is up-to-date."""

    ensure_dir(CACHE_DIR)

    src = _resolve_source_parquet()
    fp = fingerprint_file(src, schema_version=CANONICAL_SCHEMA_VERSION)

    if (
        CANONICAL_PARQUET.exists()
        and CANONICAL_META_JSON.exists()
        and cache_valid(CANONICAL_META_JSON, fp)
        and not force_rebuild
    ):
        return CANONICAL_PARQUET

    # Make sure the upstream clean parquet exists.
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
    """Convenience function used by scripts/tests."""
    ensure_canonical_parquet(force_rebuild=force_rebuild)
    return pd.read_parquet(CANONICAL_PARQUET)
