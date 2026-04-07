"""Rule-based NLP parser for basketball play-type queries."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Public data structures

@dataclass(frozen=True)
class NLPParseResult:
    """Result of parsing a natural-language situation."""
    raw_text: str
    context: Dict[str, Any]
    confidence: float
    clarifying_questions: List[str] = field(default_factory=list)
    matches: Dict[str, str] = field(default_factory=dict)

# Normalization helpers

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[,\.;]+")

def _norm(text: str) -> str:
    """Lowercase + normalize whitespace, keep ':' for clock parsing."""
    t = text.strip().lower()
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# Core parsers

_PERIOD_PATTERNS: List[Tuple[int, re.Pattern]] = [
    # Overtime
    (5, re.compile(r"\b(?:ot|overtime)\b")),
    # Q4 style
    (4, re.compile(r"\bq\s*4\b|\b4(?:th)?\s*(?:q|quarter)\b|\bfourth\s+quarter\b")),
    (3, re.compile(r"\bq\s*3\b|\b3(?:rd)?\s*(?:q|quarter)\b|\bthird\s+quarter\b")),
    (2, re.compile(r"\bq\s*2\b|\b2(?:nd)?\s*(?:q|quarter)\b|\bsecond\s+quarter\b")),
    (1, re.compile(r"\bq\s*1\b|\b1(?:st)?\s*(?:q|quarter)\b|\bfirst\s+quarter\b")),
]

def parse_period(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse quarter/period from text."""
    for period, pat in _PERIOD_PATTERNS:
        m = pat.search(text)
        if m:
            return period, m.group(0)

    # Also handle "in the 4th" without 'quarter'
    m = re.search(r"\b(1st|2nd|3rd|4th)\b", text)
    if m:
        mapping = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        return mapping.get(m.group(1)), m.group(0)

    return None, None

_TIME_MMSS_RE = re.compile(r"\b(?P<mm>\d{1,2})\s*:\s*(?P<ss>\d{2})\b")
_TIME_SECONDS_RE = re.compile(r"\b(?P<s>\d{1,3})\s*(?:s|sec|secs|second|seconds)\b")
_TIME_MINUTES_RE = re.compile(r"\b(?P<m>\d{1,2})\s*(?:m|min|mins|minute|minutes)\b")
_TIME_MIN_SEC_COMBO_RE = re.compile(
    r"\b(?P<m>\d{1,2})\s*(?:m|min|mins|minute|minutes)\s*(?P<s>\d{1,2})\s*(?:s|sec|secs|second|seconds)\b"
)

def parse_time_remaining_seconds(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse a clock reference and return *seconds remaining in the current period*."""
    # Prefer explicit mm:ss
    m = _TIME_MMSS_RE.search(text)
    if m:
        mm = int(m.group("mm"))
        ss = int(m.group("ss"))
        if 0 <= ss < 60:
            return float(mm * 60 + ss), m.group(0)

    # Prefer "X min Y sec"
    m = _TIME_MIN_SEC_COMBO_RE.search(text)
    if m:
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        if 0 <= ss < 60:
            return float(mm * 60 + ss), m.group(0)

    # "28 seconds"
    m = _TIME_SECONDS_RE.search(text)
    if m:
        s = int(m.group("s"))
        if 0 <= s <= 720:
            return float(s), m.group(0)

    # "2 minutes"
    m = _TIME_MINUTES_RE.search(text)
    if m:
        mm = int(m.group("m"))
        if 0 <= mm <= 12:
            return float(mm * 60), m.group(0)

    return None, None

# Margin patterns:
# - "down 3", "down by 3", "trailing by 3" => -3
# - "up 5", "leading by 5", "ahead by 5" => +5
# - "tie game" => 0

_MARGIN_DOWN_RE = re.compile(r"\b(?:down|trailing|behind)\s*(?:by\s*)?(?P<n>\d{1,2})\b")
_MARGIN_UP_RE = re.compile(r"\b(?:up|leading|ahead)\s*(?:by\s*)?(?P<n>\d{1,2})\b")
_MARGIN_TIED_RE = re.compile(r"\b(?:tie|tied|even)\s*(?:game|score)?\b")

def parse_score_margin(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse score margin as our_score - opp_score."""
    m = _MARGIN_TIED_RE.search(text)
    if m:
        return 0.0, m.group(0)

    m = _MARGIN_DOWN_RE.search(text)
    if m:
        n = float(m.group("n"))
        return -abs(n), m.group(0)

    m = _MARGIN_UP_RE.search(text)
    if m:
        n = float(m.group("n"))
        return abs(n), m.group(0)

    # Common basketball phrase: "one possession" / "two possessions"
    # This is ambiguous (2 vs 3), so we DON'T guess; we ask.
    if re.search(r"\b(one|two|three)\s+possession(?:s)?\b", text):
        return None, "possession(s)"

    return None, None

# Optional "extra" fields

def parse_need(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse 'need' / objective cues."""
    # Specific first
    if re.search(r"\b2\s*for\s*1\b|\btwo\s*for\s*one\b|\b2-for-1\b", text):
        return "two_for_one", "2-for-1"

    if re.search(r"\bneed\s+(?:a\s+)?(?:quick\s+)?2\b|\bquick\s+2\b", text):
        return "quick2", "quick 2"

    if re.search(r"\bneed\s+(?:a\s+)?3\b|\bneed\s+(?:a\s+)?three\b|\bneed\s+3pt\b|\bneed\s+3s\b", text):
        return "need3", "need 3"

    if re.search(r"\bneed\s+(?:a\s+)?stop\b|\bget\s+a\s+stop\b", text):
        return "stop", "stop"

    if re.search(r"\bprotect\s+the\s+ball\b|\bno\s+turnovers\b|\bsafe\b", text):
        return "safe", "safe"

    return None, None

def parse_defense_style(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse opponent defense style cues (very lightweight)."""
    if re.search(r"\bswitch(?:ing)?\b|\bswitch\s+everything\b", text):
        return "switch", "switch"

    if re.search(r"\bdrop\b|\bdrop\s+coverage\b", text):
        return "drop", "drop"

    # Zones
    m = re.search(r"\b(?:2\s*-\s*3|2\s*3)\s*zone\b|\b2\s*-\s*3\b", text)
    if m:
        return "zone_2_3", m.group(0)

    m = re.search(r"\b(?:3\s*-\s*2|3\s*2)\s*zone\b|\b3\s*-\s*2\b", text)
    if m:
        return "zone_3_2", m.group(0)

    if re.search(r"\bbox\s*-\s*and\s*-\s*1\b|\bbox\s+and\s+1\b", text):
        return "box_and_1", "box and 1"

    if re.search(r"\b1\s*-\s*3\s*-\s*1\b|\b1\s*3\s*1\b", text):
        return "zone_1_3_1", "1-3-1"

    return None, None

def parse_pace(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse pace cues: push vs slow."""
    if re.search(r"\bpush\b|\bfast\b|\bquick\b|\bearly\b|\btransition\b", text):
        return "push", "push/fast"
    if re.search(r"\bslow\b|\bwalk\s+it\s+up\b|\bburn\s+clock\b|\bmilk\s+clock\b", text):
        return "slow", "slow/burn clock"
    return None, None

# Confidence + output shaping

def _score_confidence(
    period: Optional[int],
    time_remaining: Optional[float],
    margin: Optional[float],
    extras_found: int,
) -> float:
    """Simple, explainable confidence:"""
    score = 0.0
    if period is not None:
        score += 0.34
    if time_remaining is not None:
        score += 0.34
    if margin is not None:
        score += 0.34
    score += min(0.10, extras_found * 0.02)
    return _clamp(score, 0.0, 1.0)

def _build_clarifying_questions(
    period: Optional[int],
    time_remaining: Optional[float],
    margin: Optional[float],
) -> List[str]:
    qs: List[str] = []
    if period is None:
        qs.append("What quarter/period is it? (Q1–Q4 or OT)")
    if time_remaining is None:
        qs.append("How much time is left in the quarter? (e.g., 0:28)")
    if margin is None:
        qs.append("What's the score margin? (e.g., down 3 / up 5 / tied)")
    return qs

# Main entry point

def parse_game_context(text: str, defaults: Optional[Dict[str, Any]] = None) -> NLPParseResult:
    """Parse a natural-language 'game situation' string into structured context."""
    raw = text or ""
    t = _norm(raw)

    defaults = defaults or {}

    # Core fields
    period, m_period = parse_period(t)
    time_remaining, m_time = parse_time_remaining_seconds(t)
    margin, m_margin = parse_score_margin(t)

    # Apply defaults if missing
    if period is None and "period" in defaults:
        period = int(defaults["period"])
    if time_remaining is None and "time_remaining" in defaults:
        time_remaining = _safe_float(defaults["time_remaining"])
    if margin is None and "margin" in defaults:
        margin = _safe_float(defaults["margin"])

    # Clamp where appropriate for your existing endpoint constraints.
    if time_remaining is not None:
        time_remaining = _clamp(float(time_remaining), 0.0, 720.0)
    if period is not None:
        # Accept 1-5 only (your API uses 5=OT)
        if int(period) < 1:
            period = 1
        if int(period) > 5:
            period = 5
        period = int(period)

    # Extras
    need, m_need = parse_need(t)
    defense_style, m_def = parse_defense_style(t)
    pace, m_pace = parse_pace(t)

    extras_found = sum(1 for v in [need, defense_style, pace] if v is not None)

    confidence = _score_confidence(period, time_remaining, margin, extras_found)
    clarifying = _build_clarifying_questions(period, time_remaining, margin)

    matches: Dict[str, str] = {}
    if m_period:
        matches["period"] = m_period
    if m_time:
        matches["time_remaining"] = m_time
    if m_margin:
        matches["margin"] = m_margin
    if m_need:
        matches["need"] = m_need
    if m_def:
        matches["defense_style"] = m_def
    if m_pace:
        matches["pace"] = m_pace

    context: Dict[str, Any] = {
        "period": period,
        "time_remaining": time_remaining,
        "margin": margin,
        # Optional fields (safe to ignore by existing recommenders)
        "need": need,
        "defense_style": defense_style,
        "pace": pace,
        # Keep a cleaned version of the text for UI logs/debug
        "text_normalized": t,
    }

    return NLPParseResult(
        raw_text=raw,
        context=context,
        confidence=confidence,
        clarifying_questions=clarifying,
        matches=matches,
    )

def context_to_context_ml_params(context: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a parsed `context` into the *exact* query params expected by"""
    period = context.get("period", None)
    margin = context.get("margin", None)
    time_remaining = context.get("time_remaining", None)

    if period is None or margin is None or time_remaining is None:
        missing = [k for k in ["period", "margin", "time_remaining"] if context.get(k) is None]
        raise ValueError(f"Missing required context fields: {missing}")

    return {
        "period": int(period),
        "margin": float(margin),
        "time_remaining": float(time_remaining),
    }

# Quick manual demo (optional)
