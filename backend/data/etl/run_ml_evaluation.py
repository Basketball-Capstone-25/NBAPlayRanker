"""Offline ML evaluation script for Dataset1.

Build tooling — not imported by the running FastAPI app.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from domain.baseline_recommendation import BaselineRecommender
from infrastructure.model_management.ml_models import (
    DATA_CSV_PATH,
    paired_t_test_rmse,
    pick_best_holdout_model_name,
    run_cv_evaluation,
    train_and_save_predictions_walk_forward,
)


def main() -> None:
    print("Running season-holdout evaluation (offense PPP prediction)...")

    rec = BaselineRecommender(str(DATA_CSV_PATH))

    summary, fold_metrics = run_cv_evaluation(rec.team_df, rec.league_df, n_splits=5, random_state=42)

    print("\n=== Model comparison (season-holdout) ===")
    print(summary)

    best_model = pick_best_holdout_model_name(summary)
    print(f"\nBest ML by holdout RMSE: {best_model}")

    t_stat, p_val = paired_t_test_rmse(fold_metrics, model_name="RandomForest")
    print("\nPaired t-test on fold RMSE (Baseline vs RandomForest):")
    print(f"t-statistic = {t_stat:.3f}" if t_stat is not None else "t-statistic = -- (not enough folds)")
    print(f"p-value     = {p_val:.5f}" if p_val is not None else "p-value     = -- (SciPy not installed or unavailable)")

    t2, p2 = paired_t_test_rmse(fold_metrics, model_name=best_model)
    print(f"\nPaired t-test on fold RMSE (Baseline vs {best_model}):")
    print(f"t-statistic = {t2:.3f}" if t2 is not None else "t-statistic = -- (not enough folds)")
    print(f"p-value     = {p2:.5f}" if p2 is not None else "p-value     = -- (SciPy not installed or unavailable)")

    train_and_save_predictions_walk_forward(rec.team_df, rec.league_df, model_name="auto")


if __name__ == "__main__":
    main()
