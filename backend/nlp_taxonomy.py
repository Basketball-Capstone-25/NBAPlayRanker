#backend/nlp_taxonomy.py

from __future__ import annotations

"""
Shared basketball NLP vocabulary and pattern utilities.

This file keeps the parser deterministic while still allowing broad basketball
coverage language, late-game phrases, and action-family hints.
"""

import re
from typing import Iterable, List, Optional, Sequence, Tuple


WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def num_from_token(token: str) -> Optional[int]:
    token = (token or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return WORD_NUMBERS.get(token)


def first_match(text: str, patterns: Sequence[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    for label, pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return label, m.group(0)
    return None, None


NEED_PATTERNS: List[Tuple[str, str]] = [
    ("two_for_one", r"\b(?:2\s*for\s*1|2-for-1|two\s*for\s*one)\b"),
    ("quick2", r"\b(?:need|want|looking\s+for)?\s*(?:a\s+)?(?:quick|fast|early)\s+2\b|\bquick\s+two\b"),
    ("need3", r"\b(?:need|want|get|create|hunt|looking\s+for)\s+(?:a\s+)?3(?:pt|\s*pointer)?\b|\bneed\s+three\b|\bneed\s+(?:a\s+)?three\b|\bneed\s+(?:one\s+)?three\b"),
    ("stop", r"\b(?:need|get|must\s+get)\s+(?:a\s+)?stop\b|\bdefensive\s+stop\b"),
    ("safe", r"\b(?:protect\s+the\s+ball|take\s+care\s+of\s+it|no\s+turnovers|safe\s+possession|value\s+the\s+ball)\b"),
    ("last_shot", r"\b(?:last\s+shot|hold\s+for\s+one|hold\s+for\s+the\s+last\s+shot|play\s+for\s+the\s+last\s+shot)\b"),
    ("must_score", r"\b(?:must\s+score|have\s+to\s+score|need\s+a\s+bucket|need\s+points)\b"),
    ("no_three", r"\b(?:no\s+three|take\s+away\s+the\s+three|run\s+them\s+off\s+the\s+line|don['’]t\s+give\s+up\s+a\s+3)\b"),
    ("foul_game", r"\b(?:foul\s+game|give\s+foul|foul\s+up\s+3|up\s+3\s+foul|take\s+the\s+foul)\b"),
]


SPECIAL_SITUATION_PATTERNS: List[Tuple[str, str]] = [
    ("after_timeout", r"\b(?:after\s+timeout|out\s+of\s+timeout|after\s+the\s+timeout|ato)\b"),
    ("slob", r"\b(?:slob|sideline\s+out\s+of\s+bounds|sideline\s+oob)\b"),
    ("blob", r"\b(?:blob|baseline\s+out\s+of\s+bounds|baseline\s+oob|under\s+our\s+basket)\b"),
    ("advance_ball", r"\b(?:advance\s+the\s+ball|advanced\s+it|frontcourt\s+advance)\b"),
]


DEFENSE_PATTERNS: List[Tuple[str, str]] = [
    ("switch", r"\b(?:switch(?:ing)?|switch\s+everything)\b"),
    ("drop", r"\b(?:drop|drop\s+coverage)\b"),
    ("hedge", r"\b(?:hedge|hard\s+hedge|show)\b"),
    ("blitz", r"\b(?:blitz|trap(?:ping)?\s+the\s+ball|send\s+two)\b"),
    ("ice", r"\b(?:ice|down\s+the\s+side\s+pick\s+and\s+roll)\b"),
    ("under", r"\b(?:go\s+under|under\s+the\s+screen|duck\s+under)\b"),
    ("top_lock", r"\b(?:top\s*lock|deny\s+the\s+handoff|deny\s+the\s+catch)\b"),
    ("zone_2_3", r"\b(?:2\s*-\s*3\s*zone|2\s*3\s*zone|2\s*-\s*3)\b"),
    ("zone_3_2", r"\b(?:3\s*-\s*2\s*zone|3\s*2\s*zone|3\s*-\s*2)\b"),
    ("zone_1_3_1", r"\b(?:1\s*-\s*3\s*-\s*1|1\s*3\s*1)\b"),
    ("box_and_1", r"\b(?:box\s*(?:-|\s*)and\s*(?:-|\s*)1|box\s+and\s+one)\b"),
    ("matchup_zone", r"\b(?:match\s*up\s*zone|matchup\s*zone)\b"),
]


PACE_PATTERNS: List[Tuple[str, str]] = [
    ("push", r"\b(?:push|run|fast|early\s+offense|transition|pace\s+up|play\s+fast|get\s+out\s+and\s+run)\b"),
    ("slow", r"\b(?:slow|walk\s+it\s+up|burn\s+clock|milk\s+clock|use\s+clock|slow\s+it\s+down)\b"),
]


PLAY_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    ("ball_screen", r"\b(?:pick\s*(?:and|&)\s*roll|pnr|p&r|ball\s*screen|high\s*screen|drag\s*screen|ghost\s*screen|slip\s*screen|screen\s*and\s*roll)\b"),
    ("handoff", r"\b(?:handoff|hand\s*off|dho|zoom\s*action|pistol)\b"),
    ("post_up", r"\b(?:post\s*up|post\s*touch|throw\s+it\s+inside|seal\s+inside|duck\s*in)\b"),
    ("isolation", r"\b(?:iso|isolation|clear\s*out|empty\s+side\s+iso)\b"),
    ("spot_up", r"\b(?:spot\s*up|catch\s*and\s*shoot|kick\s*out\s+three|corner\s+three|slot\s+three)\b"),
    ("off_screen", r"\b(?:off\s*screen|stagger|flare|pin\s*down|elevator|wide\s*pin|hammer)\b"),
    ("cut", r"\b(?:cut|back\s*cut|45\s*cut|dive|slip)\b"),
    ("transition", r"\b(?:transition|early\s+offense|run\s+out|hit\s+ahead)\b"),
]


PLAY_FAMILY_LABELS = {
    "ball_screen": "ball-screen action",
    "handoff": "handoff action",
    "post_up": "post touch",
    "isolation": "isolation",
    "spot_up": "spot-up shooting",
    "off_screen": "off-screen action",
    "cut": "cutting action",
    "transition": "transition push",
}


def extract_play_families(text: str) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    matches: List[str] = []
    for label, pattern in PLAY_FAMILY_PATTERNS:
        m = re.search(pattern, text)
        if m and label not in labels:
            labels.append(label)
            matches.append(m.group(0))
    return labels, matches


def family_label(family: str) -> str:
    return PLAY_FAMILY_LABELS.get(family, family.replace("_", " "))


_DEFENSE_FAMILY_MAP = {
    "switch": "switching",
    "drop": "drop coverage",
    "hedge": "hedge/show coverage",
    "blitz": "blitz/trap coverage",
    "ice": "ice/down coverage",
    "under": "under coverage",
    "top_lock": "top-lock denial",
    "zone_2_3": "2-3 zone",
    "zone_3_2": "3-2 zone",
    "zone_1_3_1": "1-3-1 zone",
    "box_and_1": "box-and-1",
    "matchup_zone": "matchup zone",
}


def defense_label(defense_style: str) -> str:
    return _DEFENSE_FAMILY_MAP.get(defense_style, defense_style.replace("_", " "))
