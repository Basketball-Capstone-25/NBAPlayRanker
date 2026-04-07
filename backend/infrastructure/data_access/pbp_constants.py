"""Play-by-play constants and paths."""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
PBP_DIR = DATA_DIR / "pbp"
CACHE_DIR = PBP_DIR / "cache"
CLEAN_PARQUET = PBP_DIR / "shots_clean.parquet"

SOURCE_PARQUET = PBP_DIR / "nba_pbp_2021_present.parquet"

ALT_SOURCE_PARQUET = DATA_DIR / "nba_pbp_2021_present.parquet"

AGG_PARQUET = PBP_DIR / "shots_agg.parquet"
AGG_LEAGUE_PARQUET = PBP_DIR / "shots_agg_league.parquet"

CANONICAL_PARQUET = CACHE_DIR / "shots_canonical.parquet"

CANONICAL_META_JSON = CACHE_DIR / "shots_canonical_meta.json"

AGG_META_JSON = CACHE_DIR / "shots_agg_meta.json"

# new features), bump this string to invalidate cached canonical files.
CANONICAL_SCHEMA_VERSION = "canonical_v1"

AGG_SCHEMA_VERSION = "agg_v1"
