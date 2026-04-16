from __future__ import annotations

"""
backend/nlp_synonyms.py

Basketball-aware synonym + normalization helpers for the NLP layer.

Purpose:
- Keep basketball vocabulary in one shared place.
- Support canonical term resolution for parser / explainer / recommender logic.
- Add optional NLTK WordNet synonym support without making the app crash if the
  corpus is missing.
- Provide deterministic text normalization so different coaching phrases map to
  the same internal concepts.

This module is intentionally import-safe:
- If NLTK WordNet is unavailable, custom basketball synonyms still work.
- No network calls are made automatically.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import nltk

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover
    wn = None  # type: ignore


SYNONYMS_VERSION = "1.0.0"

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[,;]+")

# ---------------------------------------------------------------------
# Canonical basketball vocabulary
# ---------------------------------------------------------------------

CANONICAL_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    # -------------------------
    # special situations
    # -------------------------
    "after_timeout": ("after timeout", "out of timeout", "after the timeout", "ato"),
    "slob": ("slob", "sideline out of bounds", "sideline oob"),
    "blob": ("blob", "baseline out of bounds", "baseline oob", "under our basket"),
    "advance_ball": ("advance the ball", "advanced it", "frontcourt advance"),

    # -------------------------
    # needs / late-game intent
    # -------------------------
    "quick2": (
        "quick 2",
        "quick two",
        "fast 2",
        "fast two",
        "get a quick 2",
        "get a quick two",
        "need a quick 2",
        "need a quick two",
        "quick look at the rim",
        "quick look",
    ),
    "need3": (
        "need a 3",
        "need 3",
        "need three",
        "need a three",
        "need one three",
        "hunt a 3",
        "hunt three",
        "look for a 3",
        "look for three",
        "clean 3",
        "clean three",
        "create a 3",
        "get a 3",
        "need a clean 3",
    ),
    "must_score": ("must score", "have to score", "need a bucket", "need points"),
    "last_shot": ("last shot", "hold for one", "hold for the last shot", "play for the last shot"),
    "safe": ("protect the ball", "take care of it", "no turnovers", "safe possession", "value the ball"),
    "stop": ("need a stop", "get a stop", "must get a stop", "defensive stop"),
    "no_three": (
        "no three",
        "take away the three",
        "take away the 3",
        "run them off the line",
        "dont give up a 3",
        "don't give up a 3",
    ),
    "foul_game": ("foul game", "give foul", "foul up 3", "up 3 foul", "take the foul"),
    "two_for_one": ("2 for 1", "2-for-1", "two for one"),

    # -------------------------
    # defense styles
    # -------------------------
    "switch": ("switch", "switching", "switch everything", "switching everything", "switch-heavy"),
    "drop": ("drop", "drop coverage"),
    "hedge": ("hedge", "hard hedge", "show"),
    "blitz": ("blitz", "trap the ball", "trapping the ball", "send two"),
    "ice": ("ice", "down the side pick and roll"),
    "under": ("go under", "under the screen", "duck under"),
    "top_lock": ("top lock", "top-lock", "deny the handoff", "deny the catch"),
    "zone_2_3": ("2-3 zone", "2 3 zone", "2-3"),
    "zone_3_2": ("3-2 zone", "3 2 zone", "3-2"),
    "zone_1_3_1": ("1-3-1", "1 3 1", "1-3-1 zone"),
    "box_and_1": ("box and 1", "box-and-1", "box and one"),
    "matchup_zone": ("matchup zone", "match up zone"),

    # -------------------------
    # pace intent
    # -------------------------
    "push": ("push", "run", "run out", "early offense", "pace up", "play fast", "get out and run"),
    "slow": ("slow", "slow it down", "burn clock", "milk clock", "walk it up", "use clock"),

    # -------------------------
    # play families
    # -------------------------
    "ball_screen": (
        "pick and roll",
        "pick-and-roll",
        "pick & roll",
        "pnr",
        "p&r",
        "ball screen",
        "high screen",
        "drag screen",
        "ghost screen",
        "slip screen",
        "screen and roll",
    ),
    "handoff": ("handoff", "hand off", "dho", "zoom action", "pistol"),
    "post_up": ("post up", "post-up", "post touch", "throw it inside", "seal inside", "duck in"),
    "isolation": ("iso", "isolation", "clear out", "empty side iso"),
    "spot_up": ("spot up", "spot-up", "catch and shoot", "corner three", "slot three", "kick out three"),
    "off_screen": ("off screen", "off-screen", "stagger", "flare", "pin down", "elevator", "hammer"),
    "cut": ("cut", "back cut", "45 cut", "dive", "slip cut"),
    "transition": ("transition", "early offense", "run out", "hit ahead"),
}

TERM_ENTITY_LABELS: Dict[str, str] = {
    "quick2": "NEED",
    "need3": "NEED",
    "must_score": "NEED",
    "last_shot": "NEED",
    "safe": "NEED",
    "stop": "NEED",
    "no_three": "NEED",
    "foul_game": "NEED",
    "two_for_one": "NEED",
    "after_timeout": "SPECIAL_SITUATION",
    "slob": "SPECIAL_SITUATION",
    "blob": "SPECIAL_SITUATION",
    "advance_ball": "SPECIAL_SITUATION",
    "switch": "DEFENSE_STYLE",
    "drop": "DEFENSE_STYLE",
    "hedge": "DEFENSE_STYLE",
    "blitz": "DEFENSE_STYLE",
    "ice": "DEFENSE_STYLE",
    "under": "DEFENSE_STYLE",
    "top_lock": "DEFENSE_STYLE",
    "zone_2_3": "DEFENSE_STYLE",
    "zone_3_2": "DEFENSE_STYLE",
    "zone_1_3_1": "DEFENSE_STYLE",
    "box_and_1": "DEFENSE_STYLE",
    "matchup_zone": "DEFENSE_STYLE",
    "push": "PACE_INTENT",
    "slow": "PACE_INTENT",
    "ball_screen": "PLAY_FAMILY",
    "handoff": "PLAY_FAMILY",
    "post_up": "PLAY_FAMILY",
    "isolation": "PLAY_FAMILY",
    "spot_up": "PLAY_FAMILY",
    "off_screen": "PLAY_FAMILY",
    "cut": "PLAY_FAMILY",
    "transition": "PLAY_FAMILY",
}

# Build a flat lookup map so any known alias resolves to one canonical token.
CANONICAL_LOOKUP: Dict[str, str] = {}
for canonical, variants in CANONICAL_SYNONYMS.items():
    CANONICAL_LOOKUP[canonical] = canonical
    for variant in variants:
        CANONICAL_LOOKUP[variant.lower()] = canonical


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _clean_space(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def normalize_lookup_key(text: str) -> str:
    """
    Normalize a text fragment for dictionary lookup.
    """
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _clean_space(normalized)
    return normalized


def canonicalize_term(term: str) -> str:
    """
    Resolve a raw term or phrase to its canonical basketball concept if known.
    """
    key = normalize_lookup_key(term)
    return CANONICAL_LOOKUP.get(key, key)


def get_term_label(term: str) -> Optional[str]:
    """
    Return the entity-style label for a term if it maps to a known concept.
    """
    canonical = canonicalize_term(term)
    return TERM_ENTITY_LABELS.get(canonical)


def get_custom_synonyms(term: str) -> List[str]:
    """
    Return deterministic basketball-domain synonyms for a term.
    """
    canonical = canonicalize_term(term)
    values = list(CANONICAL_SYNONYMS.get(canonical, ()))
    return dedupe_keep_order([canonical, *values])


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not value:
            continue
        key = str(value).strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


# ---------------------------------------------------------------------
# Optional NLTK / WordNet support
# ---------------------------------------------------------------------

def wordnet_available() -> bool:
    """
    Check if NLTK WordNet is available without crashing the app.
    """
    if wn is None:
        return False
    try:
        wn.ensure_loaded()
        return True
    except Exception:
        return False


def get_wordnet_synonyms(term: str) -> List[str]:
    """
    Return general-language synonyms from WordNet if available.

    Important:
    - This is a supporting feature, not the main source of basketball language.
    - Basketball-specific phrases still depend on the custom dictionary above.
    """
    if not wordnet_available():
        return []

    synonyms: Set[str] = set()
    key = normalize_lookup_key(term)
    try:
        for synset in wn.synsets(key):
            for lemma in synset.lemma_names():
                candidate = lemma.replace("_", " ").lower().strip()
                if candidate:
                    synonyms.add(candidate)
    except Exception:
        return []

    return sorted(synonyms)


def expand_synonyms(term: str, include_wordnet: bool = True) -> List[str]:
    """
    Expand a term into:
    - canonical basketball term
    - custom basketball synonyms
    - optional WordNet synonyms
    """
    custom = get_custom_synonyms(term)
    if not include_wordnet:
        return custom

    general = get_wordnet_synonyms(term)
    return dedupe_keep_order([*custom, *general])


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def build_replacement_pairs() -> List[Tuple[str, str]]:
    """
    Build alias -> canonical replacement pairs, sorted longest first so
    multi-word phrases are normalized before shorter pieces.
    """
    pairs: List[Tuple[str, str]] = []
    for canonical, variants in CANONICAL_SYNONYMS.items():
        for variant in variants:
            pairs.append((normalize_lookup_key(variant), canonical))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


REPLACEMENT_PAIRS = build_replacement_pairs()


def normalize_basketball_text(text: str) -> str:
    """
    Normalize raw coaching text into more consistent internal basketball terms.

    Example:
        "ATO, down 3, need a clean 3 vs switch-heavy coverage"
    becomes roughly:
        "after_timeout down 3 need3 vs switch coverage"
    """
    normalized = normalize_lookup_key(text)

    for alias, canonical in REPLACEMENT_PAIRS:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        normalized = pattern.sub(canonical, normalized)

    normalized = _clean_space(normalized)
    return normalized


def find_canonical_terms_in_text(text: str) -> List[str]:
    """
    Return all canonical terms detected in a text after normalization.
    """
    normalized = normalize_basketball_text(text)
    found: List[str] = []

    for canonical in CANONICAL_SYNONYMS.keys():
        pattern = re.compile(rf"(?<!\w){re.escape(canonical)}(?!\w)")
        if pattern.search(normalized):
            found.append(canonical)

    return dedupe_keep_order(found)


def group_terms_by_label(terms: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group canonical terms by their entity label.
    """
    grouped: Dict[str, List[str]] = {}
    for term in terms:
        canonical = canonicalize_term(term)
        label = TERM_ENTITY_LABELS.get(canonical)
        if not label:
            continue
        grouped.setdefault(label, []).append(canonical)

    return {label: dedupe_keep_order(values) for label, values in grouped.items()}


def extract_term_features(text: str) -> Dict[str, List[str]]:
    """
    Lightweight feature extractor based on normalized canonical basketball terms.
    Useful for parser bootstrap or testing before full spaCy entity extraction.
    """
    terms = find_canonical_terms_in_text(text)
    grouped = group_terms_by_label(terms)

    return {
        "terms": terms,
        "needs": grouped.get("NEED", []),
        "defense_styles": grouped.get("DEFENSE_STYLE", []),
        "special_situations": grouped.get("SPECIAL_SITUATION", []),
        "pace_intents": grouped.get("PACE_INTENT", []),
        "play_families": grouped.get("PLAY_FAMILY", []),
    }


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def synonyms_health() -> Dict[str, object]:
    return {
        "status": "ok",
        "synonyms_version": SYNONYMS_VERSION,
        "canonical_term_count": len(CANONICAL_SYNONYMS),
        "alias_count": sum(len(v) for v in CANONICAL_SYNONYMS.values()),
        "wordnet_available": wordnet_available(),
        "nltk_version": getattr(nltk, "__version__", None),
    }


__all__ = [
    "SYNONYMS_VERSION",
    "CANONICAL_SYNONYMS",
    "TERM_ENTITY_LABELS",
    "CANONICAL_LOOKUP",
    "normalize_lookup_key",
    "canonicalize_term",
    "get_term_label",
    "get_custom_synonyms",
    "get_wordnet_synonyms",
    "expand_synonyms",
    "normalize_basketball_text",
    "find_canonical_terms_in_text",
    "group_terms_by_label",
    "extract_term_features",
    "synonyms_health",
    "wordnet_available",
]