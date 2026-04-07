"""Recommendation services package."""
from application.recommendation_services.interfaces import (
    IGetBaselineRecommendations,
    IGetContextMLRecommendations,
)

__all__ = [
    'IGetBaselineRecommendations',
    'IGetContextMLRecommendations',
]
