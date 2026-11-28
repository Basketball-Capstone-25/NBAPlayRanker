import inspect

from domain.baseline_recommendation.baseline_recommender import rank_playtypes_baseline
from domain.context_ml_recommendation import ml_context_recommender as ml


def _defaults(fn):
    sig = inspect.signature(fn)
    return sig.parameters["w_off"].default, sig.parameters["w_def"].default


def test_baseline_defaults_to_half_offence_half_defence():
    assert _defaults(rank_playtypes_baseline) == (0.5, 0.5)


def test_context_ml_defaults_to_half_offence_half_defence():
    assert _defaults(ml.rank_ml_with_context) == (0.5, 0.5)
