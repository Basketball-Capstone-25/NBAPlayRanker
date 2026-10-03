"""Calibration arithmetic, time separation, provenance, and HTTP authorization."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.api_coordination.calibration_endpoints import create_calibration_router
from domain.statistical_analysis.calibration import compute_calibration, summarize_calibration
from infrastructure.model_management.ml_models import FEATURE_COLS


@pytest.fixture
def offense_data():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.uniform(0.1, 0.9, size=(36, len(FEATURE_COLS))), columns=FEATURE_COLS)
    frame["SEASON"] = np.repeat(["2021-22", "2022-23", "2023-24"], 12)
    frame["PLAY_TYPE"] = ["Cut", "Isolation"] * 18
    frame["POSS"] = np.repeat([100.0, 200.0, 300.0], 12)
    frame["PPP"] = 0.6 + frame["EFG_PCT"] * 0.5
    return frame


def test_signed_bias_and_error_metrics_have_ppp_units():
    report = summarize_calibration(np.array([1.1, 1.3]), np.array([1.0, 1.0]), n_bins=2)
    assert report["bias_ppp"] == pytest.approx(0.2)
    assert report["mae_ppp"] == pytest.approx(0.2)
    assert report["rmse_ppp"] == pytest.approx(np.sqrt(0.05))
    assert sum(row["count"] for row in report["bins"]) == 2


def test_perfect_predictions_and_identical_predictions():
    report = summarize_calibration(np.array([1.2, 1.2]), np.array([1.2, 1.2]))
    assert report["bias_ppp"] == report["mae_ppp"] == report["rmse_ppp"] == 0
    assert len(report["bins"]) == 1
    assert report["bins"][0]["count"] == 2
    assert report["bins"][0]["sparse"] is True


def test_empty_bins_are_explicit_and_final_upper_edge_is_included():
    report = summarize_calibration(np.array([0.5, 1.5]), np.array([1.0, 1.0]), n_bins=4)
    assert [row["count"] for row in report["bins"]] == [1, 0, 0, 1]
    assert report["bins"][1]["mean_realized_ppp"] is None
    assert report["bins"][0]["bias_ppp"] < 0
    assert report["bins"][3]["bias_ppp"] > 0


@pytest.mark.parametrize("predicted,realized", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])])
def test_invalid_calibration_inputs_are_rejected(predicted, realized):
    with pytest.raises(ValueError):
        summarize_calibration(np.array(predicted), np.array(realized))


def test_each_fold_trains_only_on_earlier_seasons(offense_data, monkeypatch):
    observed_training = []

    class ModelSpy:
        def fit(self, features, outcomes):
            observed_training.append(features[:, FEATURE_COLS.index("POSS")].tolist())
        def predict(self, features):
            return np.ones(len(features))

    monkeypatch.setattr("domain.statistical_analysis.calibration.make_pipeline", lambda *args: ModelSpy())
    report = compute_calibration(offense_data, n_splits=2)
    assert [set(values) for values in observed_training] == [{100}, {100, 200}]
    assert [fold["test_season"] for fold in report["folds"]] == ["2022-23", "2023-24"]
    assert report["summary"]["count"] == 24
    for fold in report["folds"]:
        assert max(fold["train_seasons"]) < fold["test_season"]


def test_heldout_outcomes_do_not_change_predictions_or_bins(offense_data):
    original = compute_calibration(offense_data, n_splits=1)
    changed = offense_data.copy()
    changed.loc[changed["SEASON"] == "2023-24", "PPP"] += 100
    perturbed = compute_calibration(changed, n_splits=1)
    assert original["summary"]["mean_predicted_ppp"] == perturbed["summary"]["mean_predicted_ppp"]
    assert [(b["lower_ppp"], b["upper_ppp"]) for b in original["summary"]["bins"]] == [
        (b["lower_ppp"], b["upper_ppp"]) for b in perturbed["summary"]["bins"]
    ]
    assert perturbed["summary"]["bias_ppp"] == pytest.approx(original["summary"]["bias_ppp"] - 100)


def test_global_reliability_columns_do_not_leak_into_folds(offense_data):
    original = compute_calibration(offense_data, n_splits=2)
    changed = offense_data.copy()
    changed["RELIABILITY_WEIGHT"] = 999
    changed["REL_LEAGUE"] = 999
    assert compute_calibration(changed, n_splits=2)["summary"] == original["summary"]


def test_insufficient_seasons_and_nonfinite_features_are_rejected(offense_data):
    with pytest.raises(ValueError, match="too large"):
        compute_calibration(offense_data, n_splits=5)
    with pytest.raises(ValueError, match="2 seasons"):
        compute_calibration(offense_data.iloc[:12], n_splits=1)
    offense_data.loc[0, "POSS"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_calibration(offense_data, n_splits=2)


def test_missing_schema_empty_rows_and_negative_possessions_are_rejected(offense_data):
    with pytest.raises(ValueError, match="missing columns"):
        compute_calibration(offense_data.drop(columns="PPP"), n_splits=2)
    with pytest.raises(ValueError, match="known seasons"):
        compute_calibration(offense_data.iloc[:0], n_splits=2)
    offense_data.loc[0, "POSS"] = -1
    with pytest.raises(ValueError, match="negative"):
        compute_calibration(offense_data, n_splits=2)


@pytest.fixture
def calibration_client(offense_data):
    test_app = FastAPI()
    test_app.include_router(create_calibration_router(pd.DataFrame(), pd.DataFrame()))
    with patch("application.api_coordination.calibration_endpoints.load_offense_dataset", return_value=offense_data):
        yield TestClient(test_app)


@pytest.mark.parametrize("role,expected", [("analyst", 200), ("coach", 403), (None, 403)])
def test_calibration_endpoint_requires_current_analyst_role(calibration_client, role, expected):
    service = "application.access_control_services.access_control_service"
    with patch(f"{service}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(f"{service}.get_profile_role", return_value=role):
        response = calibration_client.get("/metrics/calibration?n_splits=2", headers={"Authorization": "Bearer test"})
    assert response.status_code == expected
    if expected == 200:
        assert response.json()["evaluation"]["n_splits"] == 2


def test_calibration_endpoint_rejects_unauthenticated_and_invalid_requests(calibration_client):
    assert calibration_client.get("/metrics/calibration").status_code == 401
    service = "application.access_control_services.access_control_service"
    with patch(f"{service}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(f"{service}.get_profile_role", return_value="analyst"):
        for query in ("n_splits=0", "n_bins=0", "n_splits=5"):
            assert calibration_client.get(f"/metrics/calibration?{query}", headers={"Authorization": "Bearer test"}).status_code == 422
