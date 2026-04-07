"""Tests for context-aware ML recommendation pipeline."""
from __future__ import annotations

import pytest

from domain.context_ml_recommendation.ml_context_recommender import (
    compute_context_factors,
    label_context,
    total_time_remaining,
)


def test_total_time_remaining_start_of_game():
    assert total_time_remaining(1, 720.0) == 2880.0


def test_total_time_remaining_end_of_regulation():
    assert total_time_remaining(4, 0.0) == 0.0


def test_total_time_remaining_overtime_returns_zero():
    assert total_time_remaining(5, 120.0) == 0.0


def test_total_time_remaining_mid_third_quarter():
    result = total_time_remaining(3, 360.0)
    assert result == pytest.approx(1080.0, abs=1.0)


def test_compute_context_factors_normal():
    late, trail, lead = compute_context_factors(margin=0, period=1, time_remaining_period_sec=720)
    assert late == 0.0
    assert trail == 0.0
    assert lead == 0.0


def test_compute_context_factors_late_trailing():
    late, trail, lead = compute_context_factors(margin=-8, period=4, time_remaining_period_sec=60)
    assert late > 0.5
    assert trail > 0.0
    assert lead == 0.0


def test_compute_context_factors_blowout_leading():
    late, trail, lead = compute_context_factors(margin=20, period=4, time_remaining_period_sec=30)
    assert lead > 0.0
    assert trail == 0.0


def test_compute_context_factors_overtime():
    late, trail, lead = compute_context_factors(margin=-2, period=5, time_remaining_period_sec=60)
    assert late == 1.0  # OT → T_left=0 → late_game_factor=1
    assert trail > 0.0


def test_label_context_normal():
    assert label_context(0.0, 0.0, 0.0) == "Normal context"


def test_label_context_late_trailing():
    assert label_context(0.8, 0.5, 0.0) == "Late & trailing"


def test_label_context_late_leading():
    assert label_context(0.8, 0.0, 0.5) == "Late & leading"
