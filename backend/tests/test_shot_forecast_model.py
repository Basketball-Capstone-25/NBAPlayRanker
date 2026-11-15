import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from domain.shot_analysis.shot_forecast_features import FEATURE_COLUMNS
from domain.shot_analysis.shot_forecast_model import fit_pregame_model


@pytest.fixture
def training():
    return pd.DataFrame({
        "PRIOR_TEAM_PPS": [1.1, 0.8, 1.3], "PRIOR_OPPONENT_ALLOWED_PPS": [0.9, 1.2, 1.0],
        "PRIOR_LEAGUE_PPS": [1.0, 1.0, 1.1], "HOME_FLAG": [1, 0, 1],
        "FGA": [4, 3, 2], "POINTS": [5, 4, 0],
        "ZERO_POINT_SHOTS": [2, 1, 2], "TWO_POINT_SHOTS": [1, 2, 0], "THREE_POINT_SHOTS": [1, 0, 0],
    })


def test_weighted_fit_equals_explicit_individual_shots(training):
    grouped = fit_pregame_model(training)
    expanded_x = np.repeat(training[FEATURE_COLUMNS].to_numpy(), training.FGA, axis=0)
    expanded_y = np.array([0, 0, 2, 3, 0, 2, 2, 0, 0])
    scaler = StandardScaler().fit(expanded_x)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(expanded_x), expanded_y)
    np.testing.assert_allclose(grouped.scaler.mean_, scaler.mean_)
    np.testing.assert_allclose(grouped.scaler.scale_, scaler.scale_)
    np.testing.assert_allclose(grouped.ridge.coef_, ridge.coef_)
    expected = np.clip(ridge.predict(scaler.transform(training[FEATURE_COLUMNS].to_numpy())), 0, 3)
    np.testing.assert_allclose(grouped.predict(training), expected)


def test_prediction_ignores_current_shot_and_target_fields(training):
    model = fit_pregame_model(training)
    expected = model.predict(training)
    changed = training.assign(POINTS=999, X=999, Y=-999, SHOT_TYPE="unknown", MADE=True, MARGIN=50)
    np.testing.assert_array_equal(model.predict(changed), expected)
    assert model.parameters()["features"] == FEATURE_COLUMNS


@pytest.mark.parametrize("field,value", [("PRIOR_TEAM_PPS", np.nan), ("PRIOR_LEAGUE_PPS", 4),
    ("PRIOR_OPPONENT_ALLOWED_PPS", -1), ("HOME_FLAG", 2)])
def test_invalid_features_are_rejected(training, field, value):
    training = training.astype(float)
    training.loc[0, field] = value
    with pytest.raises(ValueError):
        fit_pregame_model(training)


def test_predictions_use_the_fixed_zero_to_three_bounds(training):
    model = fit_pregame_model(training)
    model.ridge.coef_[:] = 0
    model.ridge.intercept_ = 20
    assert np.array_equal(model.predict(training), np.full(3, 3.0))
    model.ridge.intercept_ = -20
    assert np.array_equal(model.predict(training), np.zeros(3))
