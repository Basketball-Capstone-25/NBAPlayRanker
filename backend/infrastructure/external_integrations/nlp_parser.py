from __future__ import annotations

"""
Hybrid basketball NLP parser.

What changed from the old version:
- Keeps the same public functions:
    - parse_game_context(...)
    - context_to_context_ml_params(...)
- Preserves the same main response shape for the frontend.
- Uses the new spaCy/NLTK pipeline as the primary extraction layer.
- Keeps deterministic regex/taxonomy parsing as a fallback for stability and
  backward compatibility.

Design goals:
1. Preserve current Gameplan/API behavior.
2. Stop relying on regex alone for heavy parsing.
3. Use real NLP outputs for normalization, phrase/entity extraction, and context hints.
4. Keep everything deterministic, inspectable, and defendable.
"""

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .nlp_pipeline import analyze_basketball_text  # type: ignore
    from .nlp_synonyms import canonicalize_term  # type: ignore
    from .nlp_taxonomy import (  # type: ignore
        DEFENSE_PATTERNS,
        NEED_PATTERNS,
        PACE_PATTERNS,
        SPECIAL_SITUATION_PATTERNS,
        defense_label,
        extract_play_families,
        first_match,
        num_from_token,
    )
    from .nlp_reasoning import (  # type: ignore
        build_context_warning,
        context_objective_sentence,
        derive_context_flags,
        describe_context_brief,
    )
except Exception:  # pragma: no cover
    from nlp_pipeline import analyze_basketball_text
    from nlp_synonyms import canonicalize_term
    from nlp_taxonomy import (
        DEFENSE_PATTERNS,
        NEED_PATTERNS,
        PACE_PATTERNS,
        SPECIAL_SITUATION_PATTERNS,
        defense_label,
        extract_play_families,
        first_match,
        num_from_token,
    )
    from nlp_reasoning import (
        build_context_warning,
        context_objective_sentence,
        derive_context_flags,
        describe_context_brief,
    )


PARSER_VERSION = "3.0.0"


@dataclass(frozen=True)
class NLPParseResult:
    raw_text: str
    context: Dict[str, Any]
    confidence: float
    clarifying_questions: List[str] = field(default_factory=list)
    matches: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[,;]+")


def _basic_norm(text: str) -> str:
    """
    Light normalization for deterministic fallback regex parsing.
    This intentionally does NOT do basketball synonym replacement.
    """
    t = (text or "").strip().lower()
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


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _coerce_int(value: Any, lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[int]:
    try:
        n = int(value)
    except Exception:
        return None
    if lo is not None and n < lo:
        return None
    if hi is not None and n > hi:
        return None
    return n


def _merge_unique_strings(base: Sequence[str], extra: Sequence[str]) -> List[str]:
    merged = [str(x) for x in [*base, *extra] if str(x).strip()]
    return _dedupe_keep_order(merged)


def _pick_first_match(strings: Sequence[str]) -> Optional[str]:
    for s in strings:
        if s and str(s).strip():
            return str(s)
    return None


def _parse_score_state_margin(score_state: Optional[str]) -> Optional[float]:
    if not score_state or "-" not in str(score_state):
        return None
    try:
        left, right = str(score_state).split("-", 1)
        return float(int(left) - int(right))
    except Exception:
        return None


def _entity_texts(result: Any, label: str, canonical: Optional[str] = None) -> List[str]:
    texts: List[str] = []
    if result is None:
        return texts

    for ent in getattr(result, "entities", []) or []:
        if getattr(ent, "label", None) != label:
            continue
        ent_canonical = getattr(ent, "canonical", None)
        if canonical is not None and ent_canonical != canonical:
            continue
        ent_text = getattr(ent, "text", None)
        if ent_text:
            texts.append(str(ent_text))
    return _dedupe_keep_order(texts)


def _entity_text_map_by_canonical(result: Any, label: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if result is None:
        return out

    for ent in getattr(result, "entities", []) or []:
        if getattr(ent, "label", None) != label:
            continue
        ent_canonical = getattr(ent, "canonical", None)
        ent_text = getattr(ent, "text", None)
        if ent_canonical and ent_text and ent_canonical not in out:
            out[str(ent_canonical)] = str(ent_text)
    return out


_PERIOD_PATTERNS: List[Tuple[int, re.Pattern[str]]] = [
    (5, re.compile(r"\b(?:ot|overtime|extra\s+time)\b")),
    (
        4,
        re.compile(
            r"\b(?:q\s*4|4(?:th)?\s*(?:q|quarter)|fourth\s+quarter|in\s+the\s+fourth|late\s+fourth|end\s+of\s+game|late\s+game)\b"
        ),
    ),
    (
        3,
        re.compile(
            r"\b(?:q\s*3|3(?:rd)?\s*(?:q|quarter)|third\s+quarter|in\s+the\s+third)\b"
        ),
    ),
    (
        2,
        re.compile(
            r"\b(?:q\s*2|2(?:nd)?\s*(?:q|quarter)|second\s+quarter|in\s+the\s+second)\b"
        ),
    ),
    (
        1,
        re.compile(
            r"\b(?:q\s*1|1(?:st)?\s*(?:q|quarter)|first\s+quarter|in\s+the\s+first)\b"
        ),
    ),
]


def parse_period(text: str) -> Tuple[Optional[int], Optional[str]]:
    for period, pat in _PERIOD_PATTERNS:
        m = pat.search(text)
        if m:
            return period, m.group(0)

    m = re.search(r"\b(1st|2nd|3rd|4th)\b", text)
    if m:
        mapping = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        return mapping.get(m.group(1)), m.group(0)

    return None, None


_TIME_MMSS_RE = re.compile(r"\b(?P<mm>\d{1,2})\s*:\s*(?P<ss>\d{2})\b")
_TIME_SECONDS_RE = re.compile(
    r"\b(?P<s>\d{1,3})\s*(?:s|sec|secs|second|seconds)\b"
)
_TIME_MINUTES_RE = re.compile(
    r"\b(?P<m>\d{1,2})\s*(?:m|min|mins|minute|minutes)\b"
)
_TIME_MIN_SEC_COMBO_RE = re.compile(
    r"\b(?P<m>\d{1,2})\s*(?:m|min|mins|minute|minutes)\s*(?P<s>\d{1,2})\s*(?:s|sec|secs|second|seconds)\b"
)
_UNDER_MINUTES_RE = re.compile(
    r"\bunder\s+(?P<m>\d{1,2}|one|two|three|four|five)\s+minute"
)
_LAST_SECONDS_RE = re.compile(
    r"\b(?:last|final)\s+(?P<s>\d{1,2}|ten|twenty|thirty|forty)\s+seconds\b"
)
_LESS_THAN_SECONDS_RE = re.compile(
    r"\b(?:under|less\s+than)\s+(?P<s>\d{1,2}|ten|twenty|thirty|forty)\s+seconds\b"
)
_HALF_MINUTE_RE = re.compile(r"\b(?:half\s+a\s+minute|thirty\s+seconds)\b")


def parse_time_remaining_seconds(text: str) -> Tuple[Optional[float], Optional[str]]:
    m = _TIME_MMSS_RE.search(text)
    if m:
        mm = int(m.group("mm"))
        ss = int(m.group("ss"))
        if 0 <= ss < 60:
            return float(mm * 60 + ss), m.group(0)

    m = _TIME_MIN_SEC_COMBO_RE.search(text)
    if m:
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        if 0 <= ss < 60:
            return float(mm * 60 + ss), m.group(0)

    m = _TIME_SECONDS_RE.search(text)
    if m:
        s = int(m.group("s"))
        if 0 <= s <= 720:
            return float(s), m.group(0)

    m = _TIME_MINUTES_RE.search(text)
    if m:
        mm = int(m.group("m"))
        if 0 <= mm <= 12:
            return float(mm * 60), m.group(0)

    m = _UNDER_MINUTES_RE.search(text)
    if m:
        v = num_from_token(m.group("m"))
        if v is not None and 0 <= v <= 12:
            return float(v * 60 - 1), m.group(0)

    m = _LAST_SECONDS_RE.search(text)
    if m:
        token = m.group("s")
        mapping = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40}
        s = mapping.get(token, num_from_token(token) or None)
        if s is not None:
            return float(s), m.group(0)

    m = _LESS_THAN_SECONDS_RE.search(text)
    if m:
        token = m.group("s")
        mapping = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40}
        s = mapping.get(token, num_from_token(token) or None)
        if s is not None:
            return float(max(1, s - 1)), m.group(0)

    m = _HALF_MINUTE_RE.search(text)
    if m:
        return 30.0, m.group(0)

    if re.search(r"\bfinal\s+minute\b", text):
        return 60.0, "final minute"
    if re.search(r"\bclosing\s+seconds\b", text):
        return 15.0, "closing seconds"

    return None, None


_SHOT_CLOCK_RE_1 = re.compile(
    r"\b(?P<s>\d{1,2})\s*(?:on|left\s+on)?\s+the\s+shot\s+clock\b"
)
_SHOT_CLOCK_RE_2 = re.compile(
    r"\bshot\s+clock\s+(?:at|is|under)?\s*(?P<s>\d{1,2})\b"
)
_SHOT_CLOCK_RE_3 = re.compile(
    r"\b(?P<s>\d{1,2})\s+to\s+shoot\b"
)
_SHOT_CLOCK_RE_4 = re.compile(
    r"\bwith\s+(?P<s>\d{1,2})\s+(?:seconds?\s+)?on\s+the\s+clock\b"
)


def parse_shot_clock_seconds(text: str) -> Tuple[Optional[float], Optional[str]]:
    for pat in (_SHOT_CLOCK_RE_1, _SHOT_CLOCK_RE_2, _SHOT_CLOCK_RE_3, _SHOT_CLOCK_RE_4):
        m = pat.search(text)
        if m:
            s = int(m.group("s"))
            if 0 <= s <= 24:
                return float(s), m.group(0)

    if re.search(r"\blate\s+clock\b", text):
        return 6.0, "late clock"
    if re.search(r"\bshort\s+clock\b", text):
        return 8.0, "short clock"

    return None, None


_MARGIN_DOWN_RE = re.compile(
    r"\b(?:down|trailing|behind)\s*(?:by\s*)?(?P<n>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_MARGIN_UP_RE = re.compile(
    r"\b(?:up|leading|ahead)\s*(?:by\s*)?(?P<n>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_MARGIN_TIED_RE = re.compile(r"\b(?:tie|tied|even)\s*(?:game|score)?\b")
_SCORE_PAIR_RE = re.compile(
    r"\b(?P<our>\d{2,3})\s*(?:-|/|to)\s*(?P<opp>\d{2,3})\b"
)
_ONE_POSSESSION_RE = re.compile(r"\b(one|two|three)\s+possession(?:s)?\s+game\b")
_DOWN_A_BUCKET_RE = re.compile(r"\b(?:down|trailing)\s+(?:a\s+)?bucket\b")
_UP_A_BUCKET_RE = re.compile(r"\b(?:up|leading)\s+(?:a\s+)?bucket\b")
_DOWN_ONE_SCORE_RE = re.compile(r"\b(?:down|trailing)\s+(?:one\s+)?score\b")
_UP_ONE_SCORE_RE = re.compile(r"\b(?:up|leading)\s+(?:one\s+)?score\b")


def parse_score_margin(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    m = _MARGIN_TIED_RE.search(text)
    if m:
        return 0.0, m.group(0), None

    m = _MARGIN_DOWN_RE.search(text)
    if m:
        n = num_from_token(m.group("n"))
        if n is not None:
            return -abs(float(n)), m.group(0), None

    m = _MARGIN_UP_RE.search(text)
    if m:
        n = num_from_token(m.group("n"))
        if n is not None:
            return abs(float(n)), m.group(0), None

    m = _DOWN_A_BUCKET_RE.search(text)
    if m:
        return -2.0, m.group(0), None

    m = _UP_A_BUCKET_RE.search(text)
    if m:
        return 2.0, m.group(0), None

    m = _DOWN_ONE_SCORE_RE.search(text)
    if m:
        return -2.0, m.group(0), "The phrase 'one score' was mapped conservatively to 2 points. Specify exact margin for cleaner late-game recommendations."

    m = _UP_ONE_SCORE_RE.search(text)
    if m:
        return 2.0, m.group(0), "The phrase 'one score' was mapped conservatively to 2 points. Specify exact margin for cleaner late-game recommendations."

    m = _SCORE_PAIR_RE.search(text)
    if m and re.search(r"\b(?:we[' ]?re|we\s+are|we|our|us|score)\b", text):
        our = int(m.group("our"))
        opp = int(m.group("opp"))
        return float(our - opp), m.group(0), None

    m = _ONE_POSSESSION_RE.search(text)
    if m:
        return None, m.group(0), f"The phrase '{m.group(0)}' is ambiguous. Please specify the exact score margin."

    return None, None, None


def _parse_special_situations(text: str) -> Tuple[List[str], Dict[str, str]]:
    labels: List[str] = []
    matches: Dict[str, str] = {}
    for label, pattern in SPECIAL_SITUATION_PATTERNS:
        m = re.search(pattern, text)
        if m and label not in labels:
            labels.append(label)
            matches[label] = m.group(0)

    if re.search(r"\bunder\s+our\s+basket\b", text) and "blob" not in labels:
        labels.append("blob")
        matches["blob"] = "under our basket"

    return labels, matches


def _parse_need(text: str) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    labels: List[str] = []
    matched_strings: List[str] = []

    for label, pattern in NEED_PATTERNS:
        m = re.search(pattern, text)
        if m and label not in labels:
            labels.append(label)
            matched_strings.append(m.group(0))

    manual_matches: List[Tuple[str, str]] = []

    if re.search(r"\bneed\s+(?:a\s+)?good\s+look\b", text):
        manual_matches.append(("must_score", "need a good look"))
    if re.search(r"\bmust\s+get\s+something\s+at\s+the\s+rim\b", text):
        manual_matches.append(("quick2", "something at the rim"))
    if re.search(r"\bprotect\s+the\s+lead\b", text):
        manual_matches.append(("safe", "protect the lead"))
    if re.search(r"\bmust\s+not\s+turn\s+it\s+over\b", text):
        manual_matches.append(("safe", "must not turn it over"))

    for label, matched in manual_matches:
        if label not in labels:
            labels.append(label)
            matched_strings.append(matched)

    primary = _select_primary_need(labels)

    primary_match: Optional[str] = None
    if primary is not None:
        idx = labels.index(primary)
        primary_match = matched_strings[idx]

    return primary, primary_match, labels, matched_strings


def _select_primary_need(labels: Sequence[str]) -> Optional[str]:
    priority = [
        "foul_game",
        "no_three",
        "need3",
        "quick2",
        "two_for_one",
        "last_shot",
        "stop",
        "safe",
        "must_score",
    ]
    for label in priority:
        if label in labels:
            return label
    if labels:
        return str(labels[0])
    return None


def _parse_defense_style(text: str) -> Tuple[Optional[str], Optional[str]]:
    style, matched = first_match(text, DEFENSE_PATTERNS)

    if style is None:
        if re.search(r"\bpack\s+line\b", text):
            return "gap_help", "pack line"
        if re.search(r"\bzone\b", text):
            return "generic_zone", "zone"

    return style, matched


def _parse_pace(text: str) -> Tuple[Optional[str], Optional[str]]:
    pace, matched = first_match(text, PACE_PATTERNS)

    if pace is None:
        if re.search(r"\bplay\s+through\s+the\s+clock\b", text):
            return "slow", "play through the clock"
        if re.search(r"\bget\s+it\s+up\s+quick\b", text):
            return "push", "get it up quick"

    return pace, matched


def _build_clarifying_questions(
    period: Optional[int],
    time_remaining: Optional[float],
    margin: Optional[float],
    shot_clock: Optional[float],
    ambiguity_note: Optional[str],
    defense_style: Optional[str],
    preferred_play_families: List[str],
) -> List[str]:
    questions: List[str] = []

    if period is None:
        questions.append("What quarter/period is it? (Q1–Q4 or OT)")

    if time_remaining is None:
        questions.append("How much time is left in the period? (e.g., 0:28)")

    if margin is None:
        questions.append("What is the exact score margin? (e.g., down 3 / up 5 / tied)")

    if ambiguity_note:
        questions.append(ambiguity_note)

    if shot_clock is None and time_remaining is not None and time_remaining <= 30:
        questions.append("If shot-clock timing matters, specify it explicitly (e.g., 7 on the shot clock).")

    if defense_style is None and not preferred_play_families:
        questions.append("What is the defensive look or action family? (e.g., switching, drop, zone, ball-screen, handoff)")

    return _dedupe_keep_order(questions)


def _score_confidence(
    explicit_core_hits: int,
    extras_found: int,
    defaults_used: int,
    ambiguity_penalty: float,
    vague_zone_penalty: float,
    pipeline_bonus: float,
) -> float:
    base = min(0.97, explicit_core_hits * 0.26 + extras_found * 0.05 + 0.14)
    base += pipeline_bonus
    base -= defaults_used * 0.05
    base -= ambiguity_penalty
    base -= vague_zone_penalty
    return _clamp(base, 0.0, 0.99)


def parse_game_context(text: str, defaults: Optional[Dict[str, Any]] = None) -> NLPParseResult:
    raw = text or ""
    defaults = defaults or {}

    basic_text = _basic_norm(raw)

    pipeline_result = None
    pipeline_warning: Optional[str] = None
    try:
        pipeline_result = analyze_basketball_text(raw)
    except Exception as exc:
        pipeline_warning = f"NLP pipeline fallback activated: {exc}"

    pipeline_hints: Dict[str, Any] = {}
    normalized_text = basic_text

    if pipeline_result is not None:
        pipeline_hints = dict(getattr(pipeline_result, "context_hints", {}) or {})
        normalized_text = str(getattr(pipeline_result, "normalized_text", basic_text) or basic_text)

    # ------------------------------------------------------------------
    # Primary extraction: spaCy/NLTK pipeline outputs
    # ------------------------------------------------------------------
    period = _coerce_int(pipeline_hints.get("period"), 1, 5)
    time_remaining = _safe_float(pipeline_hints.get("time_remaining"))
    shot_clock = _safe_float(pipeline_hints.get("shot_clock"))
    margin = _safe_float(pipeline_hints.get("score_margin"))
    if margin is None:
        margin = _parse_score_state_margin(pipeline_hints.get("score_state"))

    needs_from_pipeline = [
        canonicalize_term(x)
        for x in (pipeline_hints.get("needs") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    defense_candidates = [
        canonicalize_term(x)
        for x in (pipeline_hints.get("defense_styles") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    pace_candidates = [
        canonicalize_term(x)
        for x in (pipeline_hints.get("pace_intents") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    special_situations = [
        canonicalize_term(x)
        for x in (pipeline_hints.get("special_situations") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    preferred_play_families = [
        canonicalize_term(x)
        for x in (pipeline_hints.get("play_families") or [])
        if isinstance(x, str) and str(x).strip()
    ]

    need = _select_primary_need(needs_from_pipeline)
    defense_style = defense_candidates[0] if defense_candidates else None
    pace = pace_candidates[0] if pace_candidates else None

    # ------------------------------------------------------------------
    # Deterministic fallback layer for backward compatibility
    # ------------------------------------------------------------------
    fallback_period, m_period = parse_period(basic_text)
    fallback_time_remaining, m_time = parse_time_remaining_seconds(basic_text)
    fallback_shot_clock, m_shot_clock = parse_shot_clock_seconds(basic_text)
    fallback_margin, m_margin, ambiguity_note = parse_score_margin(basic_text)

    fallback_need, m_need, fallback_all_needs, all_need_matches = _parse_need(basic_text)
    fallback_defense_style, m_def = _parse_defense_style(basic_text)
    fallback_pace, m_pace = _parse_pace(basic_text)
    fallback_special_situations, special_matches = _parse_special_situations(basic_text)
    fallback_play_families, preferred_family_matches = extract_play_families(basic_text)

    if period is None:
        period = fallback_period
    if time_remaining is None:
        time_remaining = fallback_time_remaining
    if shot_clock is None:
        shot_clock = fallback_shot_clock
    if margin is None:
        margin = fallback_margin

    merged_needs = _merge_unique_strings(needs_from_pipeline, fallback_all_needs)
    need = need or fallback_need or _select_primary_need(merged_needs)
    defense_style = defense_style or fallback_defense_style
    pace = pace or fallback_pace
    special_situations = _merge_unique_strings(special_situations, fallback_special_situations)
    preferred_play_families = _merge_unique_strings(preferred_play_families, fallback_play_families)

    # ------------------------------------------------------------------
    # Defaults from UI / caller
    # ------------------------------------------------------------------
    warnings: List[str] = []
    used_defaults: List[str] = []

    if period is None and "period" in defaults:
        maybe_period = _coerce_int(defaults.get("period"), 1, 5)
        if maybe_period is not None:
            period = maybe_period
            used_defaults.append("period")

    if time_remaining is None and "time_remaining" in defaults:
        maybe_time = _safe_float(defaults.get("time_remaining"))
        if maybe_time is not None:
            time_remaining = maybe_time
            used_defaults.append("time_remaining")

    if margin is None and "margin" in defaults:
        maybe_margin = _safe_float(defaults.get("margin"))
        if maybe_margin is not None:
            margin = maybe_margin
            used_defaults.append("margin")

    if shot_clock is None and "shot_clock" in defaults:
        maybe_sc = _safe_float(defaults.get("shot_clock"))
        if maybe_sc is not None:
            shot_clock = maybe_sc
            used_defaults.append("shot_clock")

    if defense_style is None and isinstance(defaults.get("defense_style"), str):
        maybe_defense = str(defaults.get("defense_style")).strip()
        if maybe_defense:
            defense_style = canonicalize_term(maybe_defense)
            used_defaults.append("defense_style")

    if pace is None and isinstance(defaults.get("pace"), str):
        maybe_pace = str(defaults.get("pace")).strip()
        if maybe_pace:
            pace = canonicalize_term(maybe_pace)
            used_defaults.append("pace")

    if isinstance(defaults.get("special_situations"), list):
        special_situations = _merge_unique_strings(
            special_situations,
            [canonicalize_term(str(x)) for x in defaults.get("special_situations") or []],
        )

    if isinstance(defaults.get("preferred_play_families"), list):
        preferred_play_families = _merge_unique_strings(
            preferred_play_families,
            [canonicalize_term(str(x)) for x in defaults.get("preferred_play_families") or []],
        )

    merged_needs = _merge_unique_strings(merged_needs, defaults.get("needs") or [])
    need = need or _select_primary_need(merged_needs)

    # ------------------------------------------------------------------
    # Final cleanup / clamping
    # ------------------------------------------------------------------
    if period is not None:
        period = int(_clamp(float(period), 1, 5))
    if time_remaining is not None:
        time_remaining = _clamp(float(time_remaining), 0.0, 720.0)
    if shot_clock is not None:
        shot_clock = _clamp(float(shot_clock), 0.0, 24.0)
    if margin is not None:
        margin = float(_clamp(float(margin), -50.0, 50.0))

    if pipeline_warning:
        warnings.append(pipeline_warning)

    if used_defaults:
        warnings.append("Used current UI defaults for: " + ", ".join(_dedupe_keep_order(used_defaults)) + ".")

    # ------------------------------------------------------------------
    # Match strings for explainability / frontend continuity
    # ------------------------------------------------------------------
    matches: Dict[str, str] = {}

    pipeline_period_match = _pick_first_match(_entity_texts(pipeline_result, "PERIOD"))
    pipeline_time_match = _pick_first_match(_entity_texts(pipeline_result, "GAME_CLOCK"))
    pipeline_shot_clock_match = _pick_first_match(_entity_texts(pipeline_result, "SHOT_CLOCK"))
    pipeline_margin_match = _pick_first_match(_entity_texts(pipeline_result, "SCORE_MARGIN"))
    pipeline_need_matches = _entity_texts(pipeline_result, "NEED")
    pipeline_def_matches = _entity_texts(pipeline_result, "DEFENSE_STYLE")
    pipeline_pace_matches = _entity_texts(pipeline_result, "PACE_INTENT")
    pipeline_special_match_map = _entity_text_map_by_canonical(pipeline_result, "SPECIAL_SITUATION")
    pipeline_family_matches = _entity_texts(pipeline_result, "PLAY_FAMILY")

    if pipeline_period_match or m_period:
        matches["period"] = pipeline_period_match or m_period  # type: ignore[assignment]
    if pipeline_time_match or m_time:
        matches["time_remaining"] = pipeline_time_match or m_time  # type: ignore[assignment]
    if pipeline_shot_clock_match or m_shot_clock:
        matches["shot_clock"] = pipeline_shot_clock_match or m_shot_clock  # type: ignore[assignment]
    if pipeline_margin_match or m_margin:
        matches["margin"] = pipeline_margin_match or m_margin  # type: ignore[assignment]

    need_match_lookup = {canonicalize_term(x): x for x in pipeline_need_matches}
    if need and (need in need_match_lookup or m_need):
        matches["need"] = need_match_lookup.get(need) or m_need or ""
    combined_need_matches = _merge_unique_strings(pipeline_need_matches, all_need_matches)
    if combined_need_matches:
        matches["needs"] = ", ".join(combined_need_matches)

    if pipeline_def_matches or m_def:
        matches["defense_style"] = _pick_first_match(pipeline_def_matches) or m_def or ""
    if pipeline_pace_matches or m_pace:
        matches["pace"] = _pick_first_match(pipeline_pace_matches) or m_pace or ""

    merged_special_matches = dict(special_matches)
    for key, value in pipeline_special_match_map.items():
        if key not in merged_special_matches:
            merged_special_matches[key] = value
    for key, value in merged_special_matches.items():
        if value:
            matches[key] = value

    combined_family_matches = _merge_unique_strings(pipeline_family_matches, preferred_family_matches)
    if combined_family_matches:
        matches["preferred_play_families"] = ", ".join(combined_family_matches)

    # ------------------------------------------------------------------
    # Build final context
    # ------------------------------------------------------------------
    context: Dict[str, Any] = {
        "period": period,
        "time_remaining": time_remaining,
        "margin": margin,
        "need": need,
        "needs": merged_needs,
        "defense_style": defense_style,
        "pace": pace,
        "special_situations": special_situations,
        "preferred_play_families": preferred_play_families,
        "text_normalized": normalized_text,
        "raw_text": raw,
        "parser_version": PARSER_VERSION,
    }

    if shot_clock is not None:
        context["shot_clock"] = shot_clock

    # Helpful metadata for downstream logic / debugging.
    if pipeline_result is not None:
        context["nlp_pipeline_version"] = (
            getattr(pipeline_result, "metadata", {}) or {}
        ).get("pipeline_version")
        context["nlp_token_count"] = (
            getattr(pipeline_result, "metadata", {}) or {}
        ).get("token_count")

    # Derive stable downstream flags and intent tags.
    context.update(derive_context_flags(context))

    # Human-readable summaries used by Gameplan / explanations.
    context["context_brief"] = describe_context_brief(context)
    context["objective_summary"] = context_objective_sentence(context)

    warnings.extend(build_context_warning(context))

    if defense_style in {"generic_zone", "gap_help"}:
        warnings.append(
            "A broad defensive style was parsed. Specify the exact zone or coverage if possible for sharper recommendations."
        )

    if defense_style and isinstance(defense_style, str) and defense_style.startswith("zone") and not preferred_play_families:
        warnings.append(
            f"Parsed {defense_label(defense_style)}. Consider describing the action family too (ball-screen, handoff, post, spot-up, etc.) for sharper guidance."
        )

    if time_remaining is not None and shot_clock is not None and shot_clock > time_remaining:
        warnings.append(
            "Shot clock appears larger than game clock remaining. That can happen late in periods, but double-check the timing."
        )

    if need == "need3" and pace == "slow":
        warnings.append(
            "The prompt asks for a 3 while also slowing pace. That can still be valid, but the intent may be mixed."
        )

    explicit_core_hits = sum(1 for v in [period, time_remaining, margin] if v is not None)
    extras_found = (
        sum(1 for v in [need, defense_style, pace, shot_clock] if v is not None)
        + max(0, len(merged_needs) - 1)
        + len(special_situations)
        + len(preferred_play_families)
    )

    ambiguity_penalty = 0.12 if ambiguity_note else 0.0
    vague_zone_penalty = 0.08 if defense_style in {"generic_zone", "gap_help"} else 0.0
    pipeline_bonus = 0.06 if pipeline_result is not None else 0.0

    confidence = _score_confidence(
        explicit_core_hits=explicit_core_hits,
        extras_found=extras_found,
        defaults_used=len(used_defaults),
        ambiguity_penalty=ambiguity_penalty,
        vague_zone_penalty=vague_zone_penalty,
        pipeline_bonus=pipeline_bonus,
    )

    clarifying = _build_clarifying_questions(
        period=period,
        time_remaining=time_remaining,
        margin=margin,
        shot_clock=shot_clock,
        ambiguity_note=ambiguity_note,
        defense_style=defense_style,
        preferred_play_families=preferred_play_families,
    )

    return NLPParseResult(
        raw_text=raw,
        context=context,
        confidence=confidence,
        clarifying_questions=clarifying,
        matches=matches,
        warnings=_dedupe_keep_order(warnings),
    )


def context_to_context_ml_params(context: Dict[str, Any]) -> Dict[str, Any]:
    period = context.get("period")
    margin = context.get("margin")
    time_remaining = context.get("time_remaining")

    if period is None or margin is None or time_remaining is None:
        missing = [k for k in ["period", "margin", "time_remaining"] if context.get(k) is None]
        raise ValueError(f"Missing required context fields: {missing}")

    out: Dict[str, Any] = {
        "period": int(period),
        "margin": float(margin),
        "time_remaining": float(time_remaining),
    }

    shot_clock = context.get("shot_clock")
    if shot_clock is not None:
        out["shot_clock"] = float(shot_clock)

    # Preserve existing richer passthrough fields so frontend/recommender logic
    # remains backward compatible.
    passthrough_keys = [
        "need",
        "needs",
        "defense_style",
        "pace",
        "after_timeout",
        "slob",
        "blob",
        "advance_ball",
        "late_clock",
        "need3",
        "protect_lead",
        "end_of_quarter",
        "vs_switching",
        "must_stop",
        "quick2",
        "two_for_one",
        "hold_for_last",
        "foul_game",
        "no_three",
        "must_score",
        "safe",
        "special_situations",
        "preferred_play_families",
        "intent_tags",
        "offense_bias",
        "defense_bias",
        "context_brief",
        "objective_summary",
        "nlp_pipeline_version",
    ]

    for key in passthrough_keys:
        if key in context and context[key] is not None:
            out[key] = context[key]

    return out


if __name__ == "__main__":
    samples = [
        "Down 3 with 0:28 left in Q4, need a quick 2, they're switching everything",
        "Tie game, 1:45 in the 3rd, get a stop",
        "Up by 5, 2 minutes left, burn clock, vs 2-3 zone",
        "ATO, down 3, 18 on the shot clock, 0:32 in OT, need a clean 3",
        "One possession game, late in 4th, after timeout",
        "Down one bucket, 7 to shoot, need a good look vs drop",
    ]

    for sample in samples:
        result = parse_game_context(sample)
        print("-" * 100)
        print("IN :", sample)
        print("CTX:", result.context)
        print("CONF:", result.confidence)
        print("WARN:", result.warnings)
        print("ASK :", result.clarifying_questions)
        try:
            print("CTX-ML:", context_to_context_ml_params(result.context))
        except Exception as exc:
            print("CTX-ML ERROR:", exc)