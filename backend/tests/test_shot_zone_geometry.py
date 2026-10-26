import math

import numpy as np
import pandas as pd
import pytest

from domain.shot_analysis.shot_config import classify_zone, zone_from_xy, CORNER_X_MAX
from domain.shot_analysis.shot_source_quality import quarantine_score_discontinuities
from domain.shot_analysis.shot_config import normalize_shot_type


def test_long_two_is_not_an_arc_three():
    # Curry's raw event 40126680597: lateral 7, depth 22, awarded two points.
    assert zone_from_xy(22, 7, shot_value=2) == "mid"
    assert zone_from_xy(23, 3, shot_value=2) == "mid"
    assert zone_from_xy(23.5, 0) == "mid"
    assert zone_from_xy(24, 0) == "arc3"


def test_paint_is_a_rectangle_and_corner_stops_at_the_arc_join():
    assert CORNER_X_MAX == pytest.approx(math.sqrt(23.75**2 - 22**2))
    assert zone_from_xy(13, 7, shot_value=2) == "paint"
    assert zone_from_xy(13.76, 0, shot_value=2) == "mid"
    assert zone_from_xy(8, 8.01, shot_value=2) == "mid"
    assert zone_from_xy(8, 23, shot_value=3) == "corner3"
    assert zone_from_xy(10, 23, shot_value=3) == "arc3"


def test_explicit_attempt_value_settles_near_line_rounding_without_moving_coordinates():
    assert classify_zone(22.5, 0, shot_value=3) == ("arc3", "attempt_value_and_location", True)
    assert classify_zone(24, 0, shot_value=2) == ("mid", "attempt_value_and_location", True)
    assert classify_zone(23, 0, shot_value=np.nan) == ("mid", "geometry_only", True)
    assert classify_zone(26, 0, shot_value=np.nan) == ("arc3", "geometry_only", True)


def test_grossly_conflicting_or_missing_locations_are_not_fabricated():
    assert classify_zone(5, 0, shot_value=3) == ("unknown", "value_location_conflict", False)
    assert classify_zone(25, 7, shot_value=2) == ("unknown", "value_location_conflict", False)
    assert classify_zone(float("nan"), 0, shot_value=3) == ("unknown", "invalid_location", False)
    assert classify_zone(90, 0, shot_value=3) == ("unknown", "invalid_location", False)


def source_events():
    return pd.DataFrame({"game_id":["good","good","bad","bad","bad"], "game_play_number":[1,2,1,2,3],
        "id":["a","b","c","d","e"], "home_score":[2,2,2,2,2], "away_score":[0,3,0,0,2],
        "team_id":[1,2,1,1,2], "home_team_id":[1]*5, "score_value":[2,3,2,2,2],
        "shooting_play":[True]*5, "type_text":["Jump Shot"]*5})


def test_duplicate_scoring_disagreement_quarantines_whole_game_and_preserves_raw():
    raw = source_events()
    before = raw.copy(deep=True)
    retained, report = quarantine_score_discontinuities(raw)
    assert retained.game_id.tolist() == ["good", "good"]
    assert report["excluded_games"] == 1 and report["excluded_shots"] == 3
    assert report["games"][0]["inconsistent_event_ids"] == ["d"]
    pd.testing.assert_frame_equal(raw, before)


def test_checks_both_scoreboards_even_if_net_margin_is_unchanged():
    raw = source_events().iloc[:2].copy()
    raw.loc[1, ["home_score", "away_score"]] = [4,5]
    retained, report = quarantine_score_discontinuities(raw)
    assert retained.empty
    assert report["inconsistent_shot_events"] == 1


def test_spatial_model_excludes_conflicting_locations(monkeypatch):
    from domain.shot_analysis import shot_ml_models
    from test_shot_repair_contracts import complete_rows
    rows = pd.concat([complete_rows().assign(LOCATION_VALID=True), complete_rows().assign(LOCATION_VALID=False, DIST=99)], ignore_index=True)
    monkeypatch.setattr(shot_ml_models, "get_shots_clean_df", lambda: rows)
    loaded = shot_ml_models.load_shots_for_ml(max_rows=None)
    assert len(loaded) == 2 and not loaded.DIST.eq(99).any()


PROVIDER_TYPE_GROUPS = {
    "Dunk": ["Alley Oop Dunk Shot", "Cutting Dunk Shot", "Driving Dunk Shot", "Driving Reverse Dunk Shot", "Dunk Shot", "Putback Dunk Shot", "Reverse Dunk Shot", "Running Alley Oop Dunk Shot", "Running Dunk Shot", "Running Reverse Dunk Shot", "Tip Dunk Shot"],
    "Layup": ["Alley Oop Layup Shot", "Cutting Finger Roll Layup Shot", "Cutting Layup Shot", "Driving Finger Roll Layup", "Driving Layup Shot", "Finger Roll Layup", "Layup Driving Reverse", "Layup Running Reverse", "Layup Shot", "Layup Shot Putback", "Reverse Layup Shot", "Running Alley Oop Layup Shot", "Running Finger Roll Layup", "Running Layup Shot"],
    "Hook Shot": ["Driving Hook Shot", "Hook Driving Bank", "Hook Shot", "Hook Shot Bank", "Hook Turnaround Bank", "Turnaround Hook Shot"],
    "Bank Shot": ["Driving Floating Bank Jump Shot", "Driving Jump Shot Bank", "Fade Away Bank Jump Shot", "Jump Shot Bank", "Pullup Bank Jump Shot", "Step Back Bank Jump Shot", "Turnaround Bank Jump Shot", "Turnaround Fadeaway Bank Jump Shot"],
    "Fadeaway": ["Fade Away Jump Shot", "Turnaround Fade Away Jump Shot"],
    "Floater": ["Driving Floating Jump Shot", "Floating Jump Shot"],
    "Jump Shot": ["Heave Jump Shot", "Jump Shot", "Running Jump Shot", "Turnaround Jump Shot"],
    "Pullup": ["Pullup Jump Shot", "Running Pullup Jump Shot"],
    "Stepback": ["Step Back Jump Shot"],
    "Tip Shot": ["Tip Shot"],
    "Other": ["Double Personal Foul", "No Shot (Default Shot)"],
}


@pytest.mark.parametrize("source,expected", [(name,category) for category,names in PROVIDER_TYPE_GROUPS.items() for name in names])
def test_all_pinned_provider_shot_type_labels_have_reviewed_categories(source, expected):
    assert normalize_shot_type(type_text=source) == expected
