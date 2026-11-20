"""Player names for the shot data (SCRUM-825).

The cleaned shots keep only the provider's athlete ID. The raw play text names
the shooter ("Jayson Tatum makes 26-foot three point jumper"), so the name for
each ID is the one the text gives most often. Nothing is guessed: an ID whose
text never names a shooter is left out.
"""

from __future__ import annotations

import re

import pandas as pd

_SHOOTER = re.compile(r"^\s*(?P<name>[A-Z][\w.'\- ]{1,60}?)\s+(?:makes|misses)\b")


def normalize_athlete_id(value: object) -> str | None:
    """'3136193', 3136193 and 3136193.0 are the same player; blanks are None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def shooter_name(text: object) -> str | None:
    match = _SHOOTER.match(str(text or ""))
    return match.group("name").strip() if match else None


def shooter_names(rows: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """One row per athlete ID: the most common name and how many plays named it."""
    frame = pd.DataFrame({
        "athlete_id": rows[id_col].map(normalize_athlete_id),
        "name": rows[text_col].map(shooter_name),
    }).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["athlete_id", "name", "plays"])
    counts = frame.groupby(["athlete_id", "name"]).size().reset_index(name="plays")
    best = counts.sort_values(["athlete_id", "plays", "name"], ascending=[True, False, True]).drop_duplicates("athlete_id")
    return best.reset_index(drop=True)
