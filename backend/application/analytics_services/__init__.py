"""Analytics services package."""
from application.analytics_services.interfaces import (
    IGetCVEvaluation,
    IGetPairedTTest,
    IGetMLAnalysis,
    IGetShotMLAnalysis,
)

__all__ = [
    'IGetCVEvaluation',
    'IGetPairedTTest',
    'IGetMLAnalysis',
    'IGetShotMLAnalysis',
]
