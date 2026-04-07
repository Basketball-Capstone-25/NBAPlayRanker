"""Analytics service: model metrics and statistical analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from domain.baseline_recommendation import BaselineRecommender
from domain.statistical_analysis import compute_ml_analysis
from domain.shot_analysis import compute_shot_ml_analysis
from infrastructure.model_management import (
    DATA_CSV_PATH,
    paired_t_test_rmse,
    run_cv_evaluation,
)


def _build_recommender() -> BaselineRecommender:
    return BaselineRecommender(str(DATA_CSV_PATH))


def get_cv_evaluation() -> Dict[str, Any]:
    """Run cross-validation for all model families and return metrics dictionary."""
    rec = _build_recommender()
    return run_cv_evaluation(rec.team_df, rec.league_df)


def get_paired_t_test() -> Tuple[Optional[float], Optional[float]]:
    """Run paired t-test comparing RF vs baseline RMSE."""
    rec = _build_recommender()
    _, fold_metrics = run_cv_evaluation(rec.team_df, rec.league_df)
    return paired_t_test_rmse(fold_metrics)


def get_ml_analysis(season: str) -> Dict[str, Any]:
    """Compute ML statistical analysis for a given season."""
    return compute_ml_analysis(season)


def get_shot_ml_analysis(season: str) -> Dict[str, Any]:
    """Compute shot ML statistical analysis for a given season."""
    return compute_shot_ml_analysis(season)