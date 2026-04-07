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
PAINT_RADIUS = 12.0
MID_RANGE_MAX = 22.0
CORNER_Y_MIN = 22.0
CORNER_X_MAX = 14.0

def zone_from_xy(x: float, y: float, dist: Optional[float] = None) -> str:
    if dist is None:
        dist = math.sqrt((x or 0.0) ** 2 + (y or 0.0) ** 2)
    abs_y = abs(y or 0.0)
    abs_x = abs(x or 0.0)

    if dist <= RIM_RADIUS:
        return "rim"
    if dist <= PAINT_RADIUS and abs_y <= 8.0:
        return "paint"
    if dist <= MID_RANGE_MAX:
        return "mid"
    # Beyond mid-range: split into corner vs arc 3
    if abs_y >= CORNER_Y_MIN and abs_x <= CORNER_X_MAX:
        return "corner3"
    return "arc3"

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
    if "stepback" in s or "step-back" in s:
        return "Stepback"
    if "pullup" in s or "pull-up" in s:
        return "Pullup"
    if "float" in s or "runner" in s:
        return "Floater"
    if "jump" in s or "jumper" in s:
        return "Jump Shot"

    return "Other"
