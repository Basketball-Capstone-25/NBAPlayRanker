"""A fixed, attempt-weighted model for pre-game expected points per shot."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from domain.shot_analysis.shot_forecast_features import FEATURE_COLUMNS
from domain.shot_analysis.shot_forecast_metrics import summarize_shot_errors

RIDGE_ALPHA = 1.0


def _inputs(rows: pd.DataFrame) -> np.ndarray:
    values = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Forecast inputs must be non-empty and finite.")
    if (values[:, :3] < 0).any() or (values[:, :3] > 3).any() or not np.isin(values[:, 3], [0, 1]).all():
        raise ValueError("Historical points per shot must be between 0 and 3; home must be zero or one.")
    return values


@dataclass
class PregameModel:
    """Keep the scaler and Ridge fit together so later games use the same preprocessing as training."""

    scaler: StandardScaler
    ridge: Ridge

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaling and return expected points per attempt bounded between zero and three."""
        return np.clip(self.ridge.predict(self.scaler.transform(_inputs(rows))), 0, 3)

    def parameters(self) -> dict:
        """Expose the feature order, scaling, coefficients and prediction bounds used in the evaluation report."""
        return {
            "alpha": RIDGE_ALPHA, "features": FEATURE_COLUMNS,
            "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
            "coefficients": self.ridge.coef_.tolist(), "intercept": float(self.ridge.intercept_),
            "prediction_bounds": [0, 3],
        }


def fit_pregame_model(rows: pd.DataFrame) -> PregameModel:
    """Weight both scaling and fitting by attempts, matching shot-level Ridge."""
    values = _inputs(rows)
    summarize_shot_errors(rows, np.zeros(len(rows)))  # Validate target counts.
    attempts = rows.FGA.to_numpy(dtype=float)
    targets = rows.POINTS.to_numpy(dtype=float) / attempts
    scaler = StandardScaler().fit(values, sample_weight=attempts)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(values), targets, sample_weight=attempts)
    return PregameModel(scaler, ridge)
