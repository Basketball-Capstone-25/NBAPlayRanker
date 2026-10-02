"""backend/data/etl/build_pbp_pipeline.py

Phase 1: Build Dataset2 artifacts (shot attempts) end-to-end.

This script is additive and safe:
- It does NOT change Dataset1 endpoints/behavior.
- It does NOT change existing Dataset2 `/shot*` endpoints.

What it produces:
1) `backend/data/pbp/shots_clean.parquet`
   - Uppercase schema produced by `shot_etl.build_shots_dataset`.
   - Used by existing endpoints like `/viz/shot-heatmap`.
2) `backend/data/pbp/shots_agg.parquet` + `shots_agg_league.parquet`
   - Produced by `shot_aggregates.build_and_save_aggregates`.
   - Used by existing endpoint `/shotplan/rank`.
3) `backend/data/pbp/cache/shots_canonical.parquet`
   - Snake_case canonical schema produced by `pbp_clean.ensure_canonical_parquet`.
   - Used by new `/pbp/*` endpoints (Phase 2+).

Run:
  python backend/data/etl/build_pbp_pipeline.py

Optional:
  python backend/data/etl/build_pbp_pipeline.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_backend_on_path() -> Path:
    """Add backend/ to sys.path so scripts can import backend modules when run from repo root."""
    backend_dir = Path(__file__).resolve().parents[2]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    return backend_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dataset2 (PBP/shot) artifacts")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of clean + canonical cache + aggregates",
    )
    args = parser.parse_args()

    backend_dir = _ensure_backend_on_path()

    from infrastructure.data_access.pbp_constants import (
        AGG_LEAGUE_PARQUET,
        AGG_META_JSON,
        AGG_PARQUET,
        CANONICAL_PARQUET,
        CLEAN_PARQUET,
        SOURCE_PARQUET,
    )
    from infrastructure.data_access.pbp_cache import build_meta, cache_valid, fingerprint_file, write_json_atomic
    from infrastructure.data_access.pbp_constants import AGG_SCHEMA_VERSION
    from infrastructure.data_access.pbp_clean import ensure_canonical_parquet, ensure_clean_parquet
    from domain.shot_analysis.shot_aggregates import build_and_save_aggregates

    print("[build_pbp_pipeline] Starting Phase 1 build (Dataset2)")
    print(f"[build_pbp_pipeline] Backend dir: {backend_dir}")
    print(f"[build_pbp_pipeline] Source parquet expected: {SOURCE_PARQUET}")
    print(f"[build_pbp_pipeline] Clean shots parquet: {CLEAN_PARQUET}")
    print(f"[build_pbp_pipeline] Canonical parquet: {CANONICAL_PARQUET}")
    print(f"[build_pbp_pipeline] Aggregates parquet: {AGG_PARQUET}")
    print(f"[build_pbp_pipeline] League agg parquet: {AGG_LEAGUE_PARQUET}")

    # 1) Clean shots parquet (uppercase) used by existing endpoints
    ensure_clean_parquet(force_rebuild=bool(args.force))
    if not CLEAN_PARQUET.exists():
        raise RuntimeError("Failed to build shots_clean.parquet")

    # 2) Aggregates (rebuild if missing, forced, or stale)
    agg_fp = fingerprint_file(CLEAN_PARQUET, schema_version=AGG_SCHEMA_VERSION)
    agg_cache_ok = cache_valid(AGG_META_JSON, agg_fp)

    if args.force or (not AGG_PARQUET.exists()) or (not AGG_LEAGUE_PARQUET.exists()) or (not agg_cache_ok):
        reason = "--force" if args.force else "missing" if (not AGG_PARQUET.exists()) else "stale"
        print(f"[build_pbp_pipeline] Building aggregates ({reason})...")
        build_and_save_aggregates(clean_path=CLEAN_PARQUET, output_path=AGG_PARQUET)

        # Record meta so we can skip next time unless inputs change.
        meta = build_meta(
            fingerprint=agg_fp,
            extra={
                "notes": "Aggregates derived from shots_clean.parquet (shot_aggregates.py).",
                "agg_parquet": str(AGG_PARQUET),
                "league_parquet": str(AGG_LEAGUE_PARQUET),
            },
        )
        write_json_atomic(AGG_META_JSON, meta)
    else:
        print("[build_pbp_pipeline] Aggregates cache valid (skipping). Use --force to rebuild.")

    # 3) Canonical parquet (snake_case) for `/pbp/*` endpoints
    ensure_canonical_parquet(force_rebuild=bool(args.force))
    if not CANONICAL_PARQUET.exists():
        raise RuntimeError("Failed to build canonical cache parquet")

    # Summaries
    import pandas as pd

    clean_rows = len(pd.read_parquet(CLEAN_PARQUET))
    canon_rows = len(pd.read_parquet(CANONICAL_PARQUET))

    print("[build_pbp_pipeline] DONE")
    print(f"[build_pbp_pipeline] shots_clean.parquet rows: {clean_rows:,}")
    print(f"[build_pbp_pipeline] shots_canonical.parquet rows: {canon_rows:,}")
    print(f"[build_pbp_pipeline] shots_agg.parquet exists: {AGG_PARQUET.exists()}")
    print(f"[build_pbp_pipeline] shots_agg_league.parquet exists: {AGG_LEAGUE_PARQUET.exists()}")
    print(f"[build_pbp_pipeline] shots_agg_meta.json exists: {AGG_META_JSON.exists()}")


if __name__ == "__main__":
    main()
