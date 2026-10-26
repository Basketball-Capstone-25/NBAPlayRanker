from __future__ import annotations

from typing import Iterable, Optional

import math

# Season mapping: parquet season ints -> "YYYY-YY"

# Optional explicit overrides (edit if your parquet uses a nonstandard scheme).
SEASON_OVERRIDES: dict[int, str] = {}

# Most public NBA datasets encode season by the END year (e.g., 2021 -> 2020-21).
# If your parquet already uses the START year (e.g., 2021 -> 2021-22), set this to False.
SEASON_YEAR_IS_END: bool = True

def season_int_to_str(season_value: Optional[object]) -> Optional[str]:
    if season_value is None:
        return None
    try:
        season_int = int(season_value)
    except Exception:
        s = str(season_value).strip()
        return s if s else None

    if season_int in SEASON_OVERRIDES:
        return SEASON_OVERRIDES[season_int]

    if season_int < 1900:
        return str(season_int)

    if SEASON_YEAR_IS_END:
        start = season_int - 1
        end = season_int
    else:
        start = season_int
        end = season_int + 1

    return f"{start}-{str(end)[-2:]}"

# Zone mapping: (x, y) -> rim / paint / mid / corner3 / arc3

RIM_RADIUS = 4.0
PAINT_HALF_WIDTH = 8.0
PAINT_DEPTH = 13.75
THREE_POINT_RADIUS = 23.75
CORNER_Y_MIN = 22.0
CORNER_X_MAX = math.sqrt(THREE_POINT_RADIUS**2 - CORNER_Y_MIN**2)
# A quality-screen tolerance for approximate provider locations, not a claim
# about the provider's measurement precision. Conflicts beyond this are unknown.
LOCATION_VALUE_TOLERANCE = 1.5

def classify_zone(x: float, y: float, dist: Optional[float] = None, shot_value: Optional[float] = None) -> tuple[str, str, bool]:
    """Classify hoop-relative depth/lateral feet without moving the location.

    Explicit attempt values settle plausible near-line rounding disagreements.
    A serious contradiction remains unknown instead of inventing a shot location.
    Missing attempt values use geometry and retain that provenance.
    """
    if dist is None:
        dist = math.hypot(x, y)
    if not all(math.isfinite(v) for v in (x, y, dist)) or abs(y) > 25 or not -5.25 <= x <= 88.75:
        return "unknown", "invalid_location", False
    abs_y = abs(y)
    line_distance = abs_y - CORNER_Y_MIN if x <= CORNER_X_MAX else dist - THREE_POINT_RADIUS
    known_value = shot_value in (2, 3)
    if known_value and ((shot_value == 3 and line_distance < -LOCATION_VALUE_TOLERANCE) or (shot_value == 2 and line_distance > LOCATION_VALUE_TOLERANCE)):
        return "unknown", "value_location_conflict", False
    method = "attempt_value_and_location" if known_value else "geometry_only"
    is_three = shot_value == 3 if known_value else line_distance > 0
    if is_three:
        return ("corner3" if x <= CORNER_X_MAX else "arc3"), method, True
    if dist <= RIM_RADIUS:
        return "rim", method, True
    if abs_y <= PAINT_HALF_WIDTH and x <= PAINT_DEPTH:
        return "paint", method, True
    return "mid", method, True


def zone_from_xy(x: float, y: float, dist: Optional[float] = None, shot_value: Optional[float] = None) -> str:
    return classify_zone(x, y, dist, shot_value)[0]

# Shot type normalization

def _first_text(values: Iterable[Optional[str]]) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""

def normalize_shot_type(
    type_text: Optional[str] = None,
    type_abbreviation: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    raw = _first_text([type_text, type_abbreviation, text])
    if not raw:
        return "Unknown"

    s = raw.lower()

    if "dunk" in s:
        return "Dunk"
    if "layup" in s:
        return "Layup"
    if "alley" in s:
        return "Alley Oop"
    if "hook" in s:
        return "Hook Shot"
    if "tip" in s:
        return "Tip Shot"
    if "bank" in s:
        return "Bank Shot"
    if "fade" in s:
        return "Fadeaway"
    if "stepback" in s or "step-back" in s or "step back" in s:
        return "Stepback"
    if "pullup" in s or "pull-up" in s:
        return "Pullup"
    if "float" in s or "runner" in s:
        return "Floater"
    if "jump" in s or "jumper" in s:
        return "Jump Shot"

    return "Other"
