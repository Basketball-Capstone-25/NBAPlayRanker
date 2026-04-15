"""Model management interfaces."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple

import pandas as pd

class IRunCVEvaluation(Protocol):
    """Run season-holdout cross-validation for all model families."""

    def __call__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        csv_path: Path = ...,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]: ...

class IPairedTTestRMSE(Protocol):
    """Paired t-test comparing two models' fold RMSE distributions."""

    def __call__(
        self,
        fold_metrics: Dict[str, Dict[str, List[float]]],
        baseline_name: str = "Baseline (league mean)",
        model_name: str = "RandomForest",
    ) -> Tuple[float | None, float | None]: ...
