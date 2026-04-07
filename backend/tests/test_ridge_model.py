"""Tests for Ridge model evaluation."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from domain.baseline_recommendation import BaselineRecommender
from infrastructure.model_management.ml_models import (
    _make_ridge,
    FEATURE_COLS,
    RIDGE_ALPHA,
    DATA_CSV_PATH,
    load_offense_dataset,
    run_cv_evaluation,
    get_features_and_target,
)

@pytest.fixture(scope="module")
def _recommender():
    """Build BaselineRecommender once for all tests in the module."""
    return BaselineRecommender(str(DATA_CSV_PATH))

@pytest.fixture(scope="module")
def offense_data(_recommender):
    """Load offense dataset once for integration tests."""
    return load_offense_dataset(_recommender.team_df, _recommender.league_df)

@pytest.fixture(scope="module")
def cv_output(_recommender):
    """Run 5-fold season holdout once for reuse across tests."""
    return run_cv_evaluation(_recommender.team_df, _recommender.league_df, n_splits=5, random_state=42)

def test_ridge_pipeline_structure():
    """TC-ML-01 verify: pipeline has scaler then Ridge with tuned alpha."""
    pipeline = _make_ridge()

    assert isinstance(pipeline, Pipeline), (
        "Expected a sklearn Pipeline; got: " + type(pipeline).__name__
    )
    assert len(pipeline.steps) == 2, (
        f"Pipeline should have exactly 2 steps; got {len(pipeline.steps)}"
    )

    _scaler_name, scaler = pipeline.steps[0]
    _model_name, model = pipeline.steps[1]

    assert isinstance(scaler, StandardScaler), (
        "Step 0 must be StandardScaler; got: " + type(scaler).__name__
    )
    assert isinstance(model, Ridge), (
        "Step 1 must be Ridge; got: " + type(model).__name__
    )
    assert model.alpha == RIDGE_ALPHA

def test_ridge_fits_synthetic_data():
    """TC-ML-02 verify: Ridge trains on synthetic data and predicts finite values."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = len(FEATURE_COLS)

    X = rng.standard_normal((n_samples, n_features))
    y = rng.uniform(0.70, 1.30, size=n_samples)

    model = _make_ridge()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(np.isfinite(preds))
    assert preds.shape == (n_samples,)

def test_ridge_coefficients_bounded():
    """TC-ML-03 derive: Ridge coefficient norm stays under a loose bound."""
    COEF_L2_UPPER_BOUND = 50.0

    rng = np.random.default_rng(0)
    n_samples = 200
    n_features = len(FEATURE_COLS)

    X = rng.standard_normal((n_samples, n_features))
    y = rng.uniform(0.70, 1.30, size=n_samples)

    model = _make_ridge()
    model.fit(X, y)

    ridge_step = model.named_steps["model"]
    coefs = ridge_step.coef_

    assert np.all(np.isfinite(coefs))

    l2_norm = float(np.linalg.norm(coefs))
    assert l2_norm < COEF_L2_UPPER_BOUND

@pytest.mark.integration
def test_ridge_outperforms_baseline(cv_output):
    """TC-ML-04 verify: Ridge beats league-mean baseline on holdout RMSE."""
    summary_df, _ = cv_output

    ridge_rmse = float(summary_df.loc["Ridge", "RMSE_mean"])
    baseline_rmse = float(summary_df.loc["Baseline (league mean)", "RMSE_mean"])

    assert ridge_rmse < baseline_rmse

@pytest.mark.integration
def test_ridge_predictions_in_valid_range(offense_data):
    """TC-ML-05 derive: predictions stay within realistic PPP bounds."""
    PPP_MIN, PPP_MAX = 0.0, 2.5

    seasons = sorted(offense_data["SEASON"].unique())
    split_point = seasons[int(len(seasons) * 0.8)]

    train_df = offense_data[offense_data["SEASON"] < split_point]
    test_df = offense_data[offense_data["SEASON"] >= split_point]

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["PPP"].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)

    model = _make_ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert np.all(preds >= PPP_MIN)
    assert np.all(preds <= PPP_MAX)

@pytest.mark.integration
def test_ridge_per_fold_rmse_acceptable(cv_output):
    """TC-ML-06 derive: each holdout fold RMSE stays under a safety ceiling."""
    RMSE_FOLD_UPPER_BOUND = 0.50

    _, fold_metrics = cv_output
    ridge_fold_rmses = fold_metrics["Ridge"]["RMSE"]

    for fold_index, fold_rmse in enumerate(ridge_fold_rmses, start=1):
        assert fold_rmse < RMSE_FOLD_UPPER_BOUND, f"fold {fold_index} RMSE too high"

@pytest.mark.integration
def test_feature_dataset_shape(offense_data):
    """TC-ML-07 verify: feature matrix shape and finiteness are valid."""
    X, y = get_features_and_target(offense_data, FEATURE_COLS)

    assert X.ndim == 2
    assert X.shape[1] == len(FEATURE_COLS)
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))

def test_ridge_regularization_effect():
    """TC-ML-08 verify: larger alpha should shrink coefficient norm."""
    rng = np.random.default_rng(99)
    n_samples = 300
    n_features = len(FEATURE_COLS)

    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.normal(0.0, 0.5, size=n_features)
    y = X @ true_weights + rng.normal(0.0, 0.1, size=n_samples)

    def _coef_l2_norm(alpha: float) -> float:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
        pipe.fit(X, y)
        return float(np.linalg.norm(pipe.named_steps["model"].coef_))

    norm_low = _coef_l2_norm(0.001)
    norm_tuned = _coef_l2_norm(RIDGE_ALPHA)
    norm_high = _coef_l2_norm(10.0)

    assert norm_high < norm_tuned < norm_low
