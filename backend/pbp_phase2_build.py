from __future__ import annotations

import argparse
import time
from pathlib import Path

from pbp_clean import ensure_canonical_parquet, ensure_clean_parquet
from shot_aggregates import build_and_save_aggregates
from shot_ml_models import run_shot_model_cv
from shot_ml_stat_analysis import compute_shot_ml_analysis

CACHE_DIR = Path(__file__).parent / "data" / "pbp" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_CACHE = CACHE_DIR / "pbp_phase2_shot_ml_analysis.json"
CV_CACHE = CACHE_DIR / "pbp_phase2_shot_model_cv.json"
AGG_PARQUET = Path(__file__).parent / "data" / "pbp" / "shots_agg.parquet"


def main() -> None:
    import json
    import math

    parser = argparse.ArgumentParser(description="Build Dataset2 Phase 2 caches")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument(
        "--skip-phase1-rebuild",
        action="store_true",
        help="Skip forcing a rebuild of shots_clean/canonical/aggregates before Phase 2 caches.",
    )
    args = parser.parse_args()

    def sanitize(obj):
        if obj is None:
            return None
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    print("Building Phase 2 caches...")

    # IMPORTANT:
    # The original poisoned Dataset2 artifacts had MADE/POINTS all zero.
    # Rebuild Phase 1 assets first so Shot Plan / Heatmap / ML / Stats all share the same fixed data.
    if args.skip_phase1_rebuild:
        ensure_clean_parquet(force_rebuild=False)
        ensure_canonical_parquet(force_rebuild=False)
        if not AGG_PARQUET.exists():
            build_and_save_aggregates(
                clean_path=Path(__file__).parent / "data" / "pbp" / "shots_clean.parquet",
                output_path=AGG_PARQUET,
            )
    else:
        print("[phase2] Rebuilding Phase 1 shot artifacts first...")
        clean_path = ensure_clean_parquet(force_rebuild=True)
        ensure_canonical_parquet(force_rebuild=True)
        build_and_save_aggregates(clean_path=clean_path, output_path=AGG_PARQUET)

    t0 = time.time()
    analysis = compute_shot_ml_analysis(
        n_splits=int(args.n_splits),
        max_rows=int(args.max_rows),
        force_refresh=True,
    )
    analysis_out = {
        "cached": False,
        "computed_at_unix": int(time.time()),
        "compute_seconds": float(time.time() - t0),
        "payload": sanitize(analysis),
    }
    ANALYSIS_CACHE.write_text(json.dumps(analysis_out, indent=2), encoding="utf-8")
    print(f"✅ Wrote: {ANALYSIS_CACHE}")

    t1 = time.time()
    summary_df, fold_df = run_shot_model_cv(
        n_splits=int(args.n_splits),
        random_state=42,
        max_rows=min(int(args.max_rows), 75000),
    )

    metrics = []
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

    try:
        best_model = str(summary_df["RMSE_mean"].idxmin())
    except Exception:
        best_model = None

    cv_out = {
        "cached": False,
        "computed_at_unix": int(time.time()),
        "compute_seconds": float(time.time() - t1),
        "n_splits": int(args.n_splits),
        "best_model": best_model,
        "metrics": sanitize(metrics),
        "fold_summary": sanitize(fold_df.to_dict(orient="records")),
    }
    CV_CACHE.write_text(json.dumps(cv_out, indent=2), encoding="utf-8")
    print(f"✅ Wrote: {CV_CACHE}")

    print("Done.")


if __name__ == "__main__":
    main()