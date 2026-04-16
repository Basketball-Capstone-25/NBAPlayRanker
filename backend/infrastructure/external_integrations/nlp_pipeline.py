from __future__ import annotations

"""
backend/nlp_pipeline.py

spaCy-first + NLTK-supported basketball NLP utilities.

Goals:
- Give the project a real NLP layer built on established Python NLP packages.
- Keep the implementation deterministic and explainable for defense/demo use.
- Normalize coaching prompts, tokenize text, extract basketball phrases/entities,
  and surface structured hints that the parser/recommender can consume.

Notes:
- This module is intentionally import-safe. If the spaCy English model is not
  installed, it falls back to `spacy.blank("en")` so the backend still runs.
- NLTK WordNet is optional. If the corpus is unavailable, synonym expansion
  gracefully falls back to the custom basketball synonym map.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import nltk
from nltk.stem import PorterStemmer

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover
    wn = None  # type: ignore


# ---------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------

STEMMER = PorterStemmer()
PIPELINE_VERSION = "1.0.0"
DEFAULT_SPACY_MODEL = "en_core_web_sm"

_WS_RE = re.compile(r"\s+")
_SCORE_PAIR_RE = re.compile(r"\b(?P<our>\d{2,3})\s*(?:-|/|to)\s*(?P<opp>\d{2,3})\b")
_MARGIN_DOWN_RE = re.compile(
    r"\b(?:down|trailing|behind)\s*(?:by\s*)?(?P<n>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_MARGIN_UP_RE = re.compile(
    r"\b(?:up|leading|ahead)\s*(?:by\s*)?(?P<n>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_TIME_MMSS_RE = re.compile(r"\b(?P<mm>\d{1,2})\s*:\s*(?P<ss>\d{2})\b")
_SHOT_CLOCK_RE = re.compile(
    r"\b(?:(?P<a>\d{1,2})\s*(?:on|left\s+on)?\s+the\s+shot\s+clock|shot\s+clock\s+(?:at|is|under)?\s*(?P<b>\d{1,2})|(?P<c>\d{1,2})\s+to\s+shoot)\b"
)
_WORD_NUMBERS = {
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

# Custom basketball lexicon used for normalization + synonym handling.
CANONICAL_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "after_timeout": ("after timeout", "out of timeout", "after the timeout", "ato"),
    "slob": ("slob", "sideline out of bounds", "sideline oob"),
    "blob": ("blob", "baseline out of bounds", "baseline oob"),
    "advance_ball": ("advance the ball", "advanced it", "frontcourt advance"),
    "quick2": (
        "quick 2",
        "quick two",
        "fast 2",
        "fast two",
        "get a quick 2",
        "get a quick two",
        "need a quick 2",
        "need a quick two",
    ),
    "need3": (
        "need a 3",
        "need 3",
        "need three",
        "need a three",
        "hunt a 3",
        "hunt three",
        "look for a 3",
        "look for three",
        "clean 3",
        "clean three",
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
    "push": ("push", "run", "run out", "early offense", "pace up", "play fast", "get out and run"),
    "slow": ("slow", "slow it down", "burn clock", "milk clock", "walk it up", "use clock"),
    "ball_screen": (
        "pick and roll",
        "pick-and-roll",
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

ENTITY_LABELS: Dict[str, str] = {
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

PERIOD_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("5", r"\b(?:ot|overtime|extra\s+time)\b"),
    ("4", r"\b(?:q\s*4|4(?:th)?\s*(?:q|quarter)|fourth\s+quarter|late\s+fourth|late\s+game|end\s+of\s+game)\b"),
    ("3", r"\b(?:q\s*3|3(?:rd)?\s*(?:q|quarter)|third\s+quarter)\b"),
    ("2", r"\b(?:q\s*2|2(?:nd)?\s*(?:q|quarter)|second\s+quarter)\b"),
    ("1", r"\b(?:q\s*1|1(?:st)?\s*(?:q|quarter)|first\s+quarter)\b"),
)

CANONICAL_LOOKUP: Dict[str, str] = {}
for canonical, variants in CANONICAL_SYNONYMS.items():
    CANONICAL_LOOKUP[canonical] = canonical
    for variant in variants:
        CANONICAL_LOOKUP[variant.lower()] = canonical


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NLPTokenInfo:
    text: str
    lemma: str
    stem: str
    is_stop: bool
    is_alpha: bool


@dataclass(frozen=True)
class NLPEntityInfo:
    label: str
    text: str
    canonical: Optional[str]
    start_char: int
    end_char: int
    source: str


@dataclass(frozen=True)
class NLPPipelineResult:
    raw_text: str
    normalized_text: str
    tokens: List[NLPTokenInfo] = field(default_factory=list)
    entities: List[NLPEntityInfo] = field(default_factory=list)
    phrase_matches: Dict[str, List[str]] = field(default_factory=dict)
    context_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clean_space(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _num_from_token(token: str) -> Optional[int]:
    t = (token or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    return _WORD_NUMBERS.get(t)


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _safe_wordnet_synonyms(term: str) -> List[str]:
    if wn is None:
        return []
    try:
        synonyms: Set[str] = set()
        for synset in wn.synsets(term):
            for lemma in synset.lemma_names():
                candidate = lemma.replace("_", " ").lower().strip()
                if candidate:
                    synonyms.add(candidate)
        return sorted(synonyms)
    except Exception:
        return []


def canonicalize_term(term: str) -> str:
    if not term:
        return ""
    lowered = term.strip().lower()
    return CANONICAL_LOOKUP.get(lowered, lowered)


def expand_synonyms(term: str) -> List[str]:
    canonical = canonicalize_term(term)
    custom = list(CANONICAL_SYNONYMS.get(canonical, ()))
    wordnet_terms = _safe_wordnet_synonyms(term)
    return _dedupe_keep_order([canonical, *custom, *wordnet_terms])


def normalize_basketball_text(text: str) -> str:
    normalized = _clean_space(
        (text or "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .lower()
    )

    # Replace longer phrases first so short aliases do not partially rewrite text.
    replacement_pairs: List[Tuple[str, str]] = []
    for canonical, variants in CANONICAL_SYNONYMS.items():
        for variant in variants:
            replacement_pairs.append((variant.lower(), canonical))
    replacement_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for variant, canonical in replacement_pairs:
        pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)")
        normalized = pattern.sub(canonical, normalized)

    normalized = _clean_space(normalized)
    return normalized


def _build_entity_patterns() -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    for canonical, label in ENTITY_LABELS.items():
        patterns.append(
            {
                "label": label,
                "pattern": [{"LOWER": part} for part in canonical.split()],
                "id": canonical,
            }
        )
    return patterns


# ---------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------

class BasketballNLPPipeline:
    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL) -> None:
        self.model_name = model_name
        self._nlp = self._build_nlp(model_name)
        self._phrase_matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._register_phrase_patterns()

    def _build_nlp(self, model_name: str) -> Language:
        model_loaded = model_name
        try:
            nlp = spacy.load(model_name, exclude=["ner"])
        except Exception:
            nlp = spacy.blank("en")
            model_loaded = "spacy.blank('en')"

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": False})
        else:
            ruler = nlp.get_pipe("entity_ruler")

        if isinstance(ruler, EntityRuler):
            if not ruler.patterns:
                ruler.add_patterns(_build_entity_patterns())

        nlp.meta["basketball_pipeline_version"] = PIPELINE_VERSION
        nlp.meta["basketball_loaded_model"] = model_loaded
        return nlp

    def _register_phrase_patterns(self) -> None:
        for canonical, label in ENTITY_LABELS.items():
            phrase_doc = self._nlp.make_doc(canonical.replace("_", " "))
            self._phrase_matcher.add(label, [phrase_doc])
            for synonym in CANONICAL_SYNONYMS.get(canonical, ()):
                phrase_doc = self._nlp.make_doc(synonym)
                self._phrase_matcher.add(label, [phrase_doc])

    def _collect_phrase_matches(self, doc: Doc) -> Dict[str, List[str]]:
        matches = self._phrase_matcher(doc)
        grouped: Dict[str, List[str]] = {}
        for match_id, start, end in matches:
            label = self._nlp.vocab.strings[match_id]
            grouped.setdefault(label, []).append(doc[start:end].text)
        return {k: _dedupe_keep_order(v) for k, v in grouped.items()}

    def _collect_entities(self, doc: Doc, normalized_text: str) -> List[NLPEntityInfo]:
        entities: List[NLPEntityInfo] = []
        seen: Set[Tuple[str, int, int, str]] = set()

        def _push(label: str, text: str, start: int, end: int, canonical: Optional[str], source: str) -> None:
            key = (label, start, end, source)
            if key in seen:
                return
            entities.append(
                NLPEntityInfo(
                    label=label,
                    text=text,
                    canonical=canonical,
                    start_char=int(start),
                    end_char=int(end),
                    source=source,
                )
            )
            seen.add(key)

        for ent in doc.ents:
            canonical = ent.ent_id_ or canonicalize_term(ent.text)
            _push(ent.label_, ent.text, ent.start_char, ent.end_char, canonical, "spacy_entity_ruler")

        # Supplemental numeric + temporal extraction stays deterministic and transparent.
        for period_value, pattern in PERIOD_PATTERNS:
            m = re.search(pattern, normalized_text)
            if m:
                _push("PERIOD", m.group(0), m.start(), m.end(), period_value, "regex_support")
                break

        m = _TIME_MMSS_RE.search(normalized_text)
        if m:
            mm = int(m.group("mm"))
            ss = int(m.group("ss"))
            if 0 <= ss < 60:
                total_seconds = mm * 60 + ss
                _push("GAME_CLOCK", m.group(0), m.start(), m.end(), str(total_seconds), "regex_support")

        m = _SHOT_CLOCK_RE.search(normalized_text)
        if m:
            raw = m.group("a") or m.group("b") or m.group("c")
            if raw is not None:
                shot_clock = int(raw)
                if 0 <= shot_clock <= 24:
                    _push("SHOT_CLOCK", m.group(0), m.start(), m.end(), str(shot_clock), "regex_support")

        m = _MARGIN_DOWN_RE.search(normalized_text)
        if m:
            n = _num_from_token(m.group("n"))
            if n is not None:
                _push("SCORE_MARGIN", m.group(0), m.start(), m.end(), str(-abs(n)), "regex_support")
        else:
            m = _MARGIN_UP_RE.search(normalized_text)
            if m:
                n = _num_from_token(m.group("n"))
                if n is not None:
                    _push("SCORE_MARGIN", m.group(0), m.start(), m.end(), str(abs(n)), "regex_support")

        m = _SCORE_PAIR_RE.search(normalized_text)
        if m:
            our = int(m.group("our"))
            opp = int(m.group("opp"))
            _push("SCORE_STATE", m.group(0), m.start(), m.end(), f"{our}-{opp}", "regex_support")

        # Phrase matcher can surface extra domain mentions that the ruler may not catch.
        for label, phrases in self._collect_phrase_matches(doc).items():
            for phrase in phrases:
                canonical = canonicalize_term(phrase)
                start = normalized_text.find(phrase.lower())
                end = start + len(phrase) if start >= 0 else 0
                _push(label, phrase, max(0, start), max(0, end), canonical, "spacy_phrase_matcher")

        entities.sort(key=lambda item: (item.start_char, item.end_char, item.label, item.source))
        return entities

    def _build_context_hints(self, entities: Sequence[NLPEntityInfo]) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "needs": [],
            "defense_styles": [],
            "special_situations": [],
            "play_families": [],
            "pace_intents": [],
            "period": None,
            "time_remaining": None,
            "shot_clock": None,
            "score_margin": None,
            "score_state": None,
        }

        for ent in entities:
            value = ent.canonical or ent.text
            if ent.label == "NEED":
                hints["needs"].append(value)
            elif ent.label == "DEFENSE_STYLE":
                hints["defense_styles"].append(value)
            elif ent.label == "SPECIAL_SITUATION":
                hints["special_situations"].append(value)
            elif ent.label == "PLAY_FAMILY":
                hints["play_families"].append(value)
            elif ent.label == "PACE_INTENT":
                hints["pace_intents"].append(value)
            elif ent.label == "PERIOD" and hints["period"] is None:
                try:
                    hints["period"] = int(value)
                except Exception:
                    hints["period"] = value
            elif ent.label == "GAME_CLOCK" and hints["time_remaining"] is None:
                try:
                    hints["time_remaining"] = float(value)
                except Exception:
                    hints["time_remaining"] = value
            elif ent.label == "SHOT_CLOCK" and hints["shot_clock"] is None:
                try:
                    hints["shot_clock"] = float(value)
                except Exception:
                    hints["shot_clock"] = value
            elif ent.label == "SCORE_MARGIN" and hints["score_margin"] is None:
                try:
                    hints["score_margin"] = float(value)
                except Exception:
                    hints["score_margin"] = value
            elif ent.label == "SCORE_STATE" and hints["score_state"] is None:
                hints["score_state"] = value

        for key in ("needs", "defense_styles", "special_situations", "play_families", "pace_intents"):
            hints[key] = _dedupe_keep_order([str(v) for v in hints[key]])

        # Convenience booleans for downstream parser/recommender wiring.
        for need in hints["needs"]:
            hints[need] = True
        for special in hints["special_situations"]:
            hints[special] = True
        for style in hints["defense_styles"]:
            hints[f"vs_{style}"] = True
        for family in hints["play_families"]:
            hints.setdefault("preferred_play_families", []).append(family)
        hints["preferred_play_families"] = _dedupe_keep_order(hints.get("preferred_play_families", []))

        return hints

    def parse(self, text: str) -> NLPPipelineResult:
        raw_text = text or ""
        normalized_text = normalize_basketball_text(raw_text)
        doc = self._nlp(normalized_text)

        tokens: List[NLPTokenInfo] = []
        for token in doc:
            lemma = token.lemma_ if token.lemma_ else token.text.lower()
            tokens.append(
                NLPTokenInfo(
                    text=token.text,
                    lemma=lemma,
                    stem=STEMMER.stem(token.text.lower()),
                    is_stop=bool(token.is_stop),
                    is_alpha=bool(token.is_alpha),
                )
            )

        entities = self._collect_entities(doc, normalized_text)
        phrase_matches = self._collect_phrase_matches(doc)
        context_hints = self._build_context_hints(entities)

        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "spacy_model_requested": self.model_name,
            "spacy_model_loaded": self._nlp.meta.get("basketball_loaded_model", self.model_name),
            "spacy_pipe_names": list(self._nlp.pipe_names),
            "nltk_version": getattr(nltk, "__version__", None),
            "wordnet_available": bool(_safe_wordnet_synonyms("fast")) if wn is not None else False,
            "token_count": len(tokens),
            "sentence_count": sum(1 for _ in doc.sents),
        }

        return NLPPipelineResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            tokens=tokens,
            entities=entities,
            phrase_matches=phrase_matches,
            context_hints=context_hints,
            metadata=metadata,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_version": PIPELINE_VERSION,
            "spacy_model_requested": self.model_name,
            "spacy_model_loaded": self._nlp.meta.get("basketball_loaded_model", self.model_name),
            "spacy_pipe_names": list(self._nlp.pipe_names),
            "nltk_version": getattr(nltk, "__version__", None),
            "wordnet_available": bool(_safe_wordnet_synonyms("fast")) if wn is not None else False,
        }


# ---------------------------------------------------------------------
# Shared singleton helpers
# ---------------------------------------------------------------------

_PIPELINE_SINGLETON: Optional[BasketballNLPPipeline] = None


def get_nlp_pipeline() -> BasketballNLPPipeline:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = BasketballNLPPipeline()
    return _PIPELINE_SINGLETON


def analyze_basketball_text(text: str) -> NLPPipelineResult:
    return get_nlp_pipeline().parse(text)


def pipeline_health() -> Dict[str, Any]:
    return get_nlp_pipeline().health()


__all__ = [
    "BasketballNLPPipeline",
    "NLPTokenInfo",
    "NLPEntityInfo",
    "NLPPipelineResult",
    "PIPELINE_VERSION",
    "DEFAULT_SPACY_MODEL",
    "CANONICAL_SYNONYMS",
    "canonicalize_term",
    "expand_synonyms",
    "normalize_basketball_text",
    "get_nlp_pipeline",
    "analyze_basketball_text",
    "pipeline_health",
]