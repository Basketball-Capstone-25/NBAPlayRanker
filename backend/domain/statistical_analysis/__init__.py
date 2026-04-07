"""Statistical analysis subsystem."""
from domain.statistical_analysis.interfaces import IComputeMLAnalysis
from domain.statistical_analysis.ml_stat_analysis import compute_ml_analysis

__all__ = [
    "IComputeMLAnalysis",
    "compute_ml_analysis",
]
