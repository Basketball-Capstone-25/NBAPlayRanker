"""Rebuild derived play-type artifacts after a source or aggregation change.

Run offline from the repository root with backend/.venv/bin/python. The raw
Synergy source and all shot datasets remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from threadpoolctl import threadpool_limits

from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION, SEASON_PHASE
from domain.statistical_analysis import ml_stat_analysis
from domain.statistical_analysis.forecast_evaluation import evaluate_forecasts
from infrastructure.model_management.ml_models import (
    pick_best_holdout_model_name, run_cv_evaluation,
    train_and_save_predictions_walk_forward, validate_prediction_provenance,
)
from infrastructure.data_access.dataset_metadata import read_versioned_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BACKEND / "data/synergy_playtypes_2019_2025_players.csv")
    parser.add_argument("--output-dir", type=Path, default=BACKEND / "data")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rec = BaselineRecommender(str(args.source))
    with threadpool_limits(limits=1):
        summary, folds = run_cv_evaluation(rec.team_df, rec.league_df, n_splits=5, random_state=42)
        selected = pick_best_holdout_model_name(summary)
        target = args.output_dir / "ml_offense_ppp_predictions.csv"
        train_and_save_predictions_walk_forward(rec.team_df, rec.league_df, output_path=target, model_name=selected, random_state=42)
        validate_prediction_provenance(read_versioned_csv(target), rec.team_df)
        ml_stat_analysis.ANALYSIS_CACHE_PATH = args.output_dir / "cache/ml_analysis_cache.json"
        analysis = ml_stat_analysis.compute_ml_analysis(rec.team_df, rec.league_df, n_splits=5, min_poss=25, force_refresh=True)
        forecast, predictions = evaluate_forecasts(rec.raw_df)
    forecast["source_dataset_version"] = rec.raw_df.attrs["dataset_version"]
    forecast["aggregation_version"] = AGGREGATION_VERSION
    (args.evidence_dir / "forecast-report.json").write_text(json.dumps(forecast, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(args.evidence_dir / "forecast-predictions.csv", index=False)
    receipt = {
        "source_dataset_version": rec.raw_df.attrs["dataset_version"],
        "aggregation_version": AGGREGATION_VERSION, "season_phase": SEASON_PHASE,
        "source_rows": len(rec.raw_df), "team_rows": len(rec.team_df),
        "selected_model": selected, "holdout_summary": summary.reset_index().to_dict(orient="records"),
        "holdout_fold_metrics": folds,
        "prediction_rows": len(read_versioned_csv(target)),
        "prediction_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "analysis_cache_sha256": hashlib.sha256(ml_stat_analysis.ANALYSIS_CACHE_PATH.read_bytes()).hexdigest(),
        "analysis_rows": analysis["dataset"]["rows_after_filters"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope": "Offline regular-season diagnostics use observed-season rate features. The separate forecast report uses prior-season features; neither establishes a service or accuracy guarantee.",
    }
    (args.evidence_dir / "artifact-rebuild.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"Rebuilt {receipt['prediction_rows']} play-type predictions and the source-matched analysis cache in {receipt['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
