from __future__ import annotations

"""
Derived basketball reasoning utilities for the deterministic NLP layer.

This module is intentionally:
- deterministic
- lightweight
- import-safe for both `backend.*` and direct script execution

It converts parsed coaching language into stable, basketball-specific helper
signals that the parser, explainer, and Gameplan flow can all reuse.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .nlp_taxonomy import family_label  # type: ignore
except Exception:  # pragma: no cover
    from nlp_taxonomy import family_label


PLAY_FAMILY_ALIASES: Dict[str, Sequence[str]] = {
    "ball_screen": (
        "pick",
        "pick and roll",
        "pick-and-roll",
        "screen and roll",
        "screen-and-roll",
        "ball screen",
        "ball-screen",
        "p&r",
        "pnr",
        "drag",
        "ghost",
        "slip",
        "high screen",
        "middle pick",
    ),
    "handoff": (
        "handoff",
        "hand off",
        "dho",
        "zoom",
        "pistol",
        "chase handoff",
    ),
    "post_up": (
        "post",
        "post up",
        "post-up",
        "duck in",
        "duck-in",
        "seal",
        "seal inside",
        "touch inside",
    ),
    "isolation": (
        "iso",
        "isolation",
        "clear out",
        "clear-out",
        "empty side iso",
    ),
    "spot_up": (
        "spot up",
        "spot-up",
        "catch and shoot",
        "catch-and-shoot",
        "corner three",
        "slot three",
        "kick out",
        "kick-out",
    ),
    "off_screen": (
        "off screen",
        "off-screen",
        "stagger",
        "flare",
        "pin down",
        "pin-down",
        "elevator",
        "hammer",
        "wide pin",
    ),
    "cut": (
        "cut",
        "back cut",
        "back-cut",
        "45 cut",
        "45-cut",
        "dive",
        "slip",
        "slip cut",
    ),
    "transition": (
        "transition",
        "early offense",
        "early-offense",
        "run out",
        "run-out",
        "hit ahead",
        "push",
        "run",
    ),
}


_SPECIAL_LABELS = {
    "after_timeout": "after timeout",
    "slob": "sideline out of bounds",
    "blob": "baseline out of bounds",
    "advance_ball": "advanced-ball situation",
}


NEED_LABELS = {
    "two_for_one": "2-for-1",
    "quick2": "quick 2",
    "need3": "need a 3",
    "stop": "need a stop",
    "safe": "protect the ball",
    "last_shot": "hold for last shot",
    "must_score": "must score",
    "no_three": "take away the 3",
    "foul_game": "foul game",
}


DEFENSE_STYLE_LABELS = {
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


PACE_LABELS = {
    "push": "push pace",
    "slow": "slow it down",
}


TRUE_NEED_KEYS = {
    "two_for_one",
    "quick2",
    "need3",
    "stop",
    "safe",
    "last_shot",
    "must_score",
    "no_three",
    "foul_game",
}


SPECIAL_KEYS = {"after_timeout", "slob", "blob", "advance_ball"}


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def infer_play_family_from_name(play_name: str) -> Optional[str]:
    name = _clean_text(play_name)
    if not name:
        return None

    ordered_families: List[str] = [
        "ball_screen",
        "handoff",
        "post_up",
        "isolation",
        "spot_up",
        "off_screen",
        "cut",
        "transition",
    ]

    for family in ordered_families:
        aliases = PLAY_FAMILY_ALIASES.get(family, ())
        for alias in aliases:
            if _clean_text(alias) in name:
                return family
    return None


def need_label(need: Optional[str]) -> Optional[str]:
    if not need:
        return None
    return NEED_LABELS.get(need, str(need).replace("_", " "))


def defense_style_label(defense_style: Optional[str]) -> Optional[str]:
    if not defense_style:
        return None
    return DEFENSE_STYLE_LABELS.get(defense_style, str(defense_style).replace("_", " "))


def pace_label(pace: Optional[str]) -> Optional[str]:
    if not pace:
        return None
    return PACE_LABELS.get(pace, str(pace).replace("_", " "))


def special_situation_labels(context: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for code in list(context.get("special_situations") or []):
        labels.append(_SPECIAL_LABELS.get(code, str(code).replace("_", " ")))
    return _dedupe_keep_order(labels)


def family_fit_sentences(requested_families: List[str], play_name: str) -> List[str]:
    out: List[str] = []
    play_family = infer_play_family_from_name(play_name)
    requested = _dedupe_keep_order([str(x) for x in requested_families or [] if x])
    if not play_family or not requested:
        return out

    if play_family in requested:
        out.append(
            f"Intent fit: the prompt points toward {family_label(play_family)}, and this play matches that family."
        )
    else:
        requested_labels = ", ".join(family_label(f) for f in requested)
        out.append(
            f"Intent note: this play reads as {family_label(play_family)}, while the prompt leaned toward {requested_labels}."
        )
    return out


def _normalized_needs(context: Dict[str, Any]) -> List[str]:
    raw_needs = list(context.get("needs") or [])
    single_need = context.get("need")
    if isinstance(single_need, str) and single_need:
        raw_needs.append(single_need)

    for key in TRUE_NEED_KEYS:
        if bool(context.get(key)):
            raw_needs.append(key)

    return _dedupe_keep_order([str(x) for x in raw_needs if isinstance(x, str) and x])


def _normalized_specials(context: Dict[str, Any]) -> List[str]:
    raw_specials = list(context.get("special_situations") or [])
    for key in SPECIAL_KEYS:
        if bool(context.get(key)):
            raw_specials.append(key)
    return _dedupe_keep_order([str(x) for x in raw_specials if isinstance(x, str) and x])


def derive_context_flags(context: Dict[str, Any]) -> Dict[str, Any]:
    period = context.get("period")
    time_remaining = _to_float(context.get("time_remaining"))
    margin = _to_float(context.get("margin"))
    shot_clock = _to_float(context.get("shot_clock"))

    needs = set(_normalized_needs(context))
    specials = set(_normalized_specials(context))

    defense_style = context.get("defense_style")
    pace = context.get("pace")

    after_timeout = "after_timeout" in specials
    slob = "slob" in specials
    blob = "blob" in specials
    advance_ball = "advance_ball" in specials

    quick2 = "quick2" in needs
    need3 = "need3" in needs
    two_for_one = "two_for_one" in needs
    must_stop = "stop" in needs
    hold_for_last = "last_shot" in needs
    foul_game = "foul_game" in needs
    no_three = "no_three" in needs
    must_score = "must_score" in needs
    safe = "safe" in needs

    protect_lead = bool(context.get("protect_lead"))
    if not protect_lead and margin is not None and margin > 0:
        if time_remaining is not None and time_remaining <= 90:
            protect_lead = True
        if no_three or foul_game:
            protect_lead = True

    end_of_quarter = bool(context.get("end_of_quarter"))
    if not end_of_quarter:
        if time_remaining is not None and time_remaining <= 45:
            end_of_quarter = True
        if hold_for_last:
            end_of_quarter = True

    late_clock = bool(context.get("late_clock"))
    if not late_clock:
        if time_remaining is not None and time_remaining <= 30:
            late_clock = True
        if shot_clock is not None and shot_clock <= 8:
            late_clock = True

    vs_switching = bool(context.get("vs_switching")) or defense_style == "switch"

    offense_bias = 0.50
    defense_bias = 0.50

    if must_stop or no_three:
        offense_bias = 0.20
        defense_bias = 0.80
    elif need3 or quick2 or two_for_one or must_score or (margin is not None and margin < 0):
        offense_bias = 0.75
        defense_bias = 0.25
    elif protect_lead or safe:
        offense_bias = 0.45
        defense_bias = 0.55

    preferred_families = _dedupe_keep_order(
        [str(x) for x in list(context.get("preferred_play_families") or []) if x]
    )

    intent_tags: List[str] = []
    intent_tags.extend([f"need:{n}" for n in sorted(needs)])
    intent_tags.extend([f"special:{s}" for s in sorted(specials)])
    if isinstance(defense_style, str) and defense_style:
        intent_tags.append(f"defense:{defense_style}")
    if isinstance(pace, str) and pace:
        intent_tags.append(f"pace:{pace}")
    intent_tags.extend([f"family:{fam}" for fam in preferred_families])
    if late_clock:
        intent_tags.append("tempo:late_clock")
    if protect_lead:
        intent_tags.append("score:protect_lead")

    out: Dict[str, Any] = {
        "after_timeout": after_timeout,
        "slob": slob,
        "blob": blob,
        "advance_ball": advance_ball,
        "late_clock": late_clock,
        "need3": need3,
        "protect_lead": protect_lead,
        "end_of_quarter": end_of_quarter,
        "vs_switching": vs_switching,
        "must_stop": must_stop,
        "quick2": quick2,
        "two_for_one": two_for_one,
        "hold_for_last": hold_for_last,
        "foul_game": foul_game,
        "no_three": no_three,
        "must_score": must_score,
        "safe": safe,
        "offense_bias": round(float(offense_bias), 2),
        "defense_bias": round(float(defense_bias), 2),
        "needs": sorted(needs),
        "special_situations": sorted(specials),
        "preferred_play_families": preferred_families,
        "intent_tags": _dedupe_keep_order(intent_tags),
    }

    if shot_clock is not None:
        out["shot_clock"] = shot_clock

    if isinstance(period, int):
        out["period"] = period
    if time_remaining is not None:
        out["time_remaining"] = time_remaining
    if margin is not None:
        out["margin"] = margin
    if isinstance(defense_style, str) and defense_style:
        out["defense_style"] = defense_style
    if isinstance(pace, str) and pace:
        out["pace"] = pace

    return out


def build_context_warning(context: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    period = context.get("period")
    margin = _to_float(context.get("margin"))
    time_remaining = _to_float(context.get("time_remaining"))
    shot_clock = _to_float(context.get("shot_clock"))
    defense_style = context.get("defense_style")

    if period == 5:
        warnings.append(
            "Parsed overtime as period 5. Make sure your UI and ranking flow preserve OT instead of forcing quarter 4."
        )

    if time_remaining is not None and time_remaining <= 45 and shot_clock is None:
        warnings.append(
            "Late-game clock was parsed, but no explicit shot-clock value was found. The urgency is usable, but a real shot clock would sharpen the recommendation."
        )

    if (
        margin is not None
        and abs(float(margin)) >= 15
        and time_remaining is not None
        and time_remaining <= 30
    ):
        warnings.append(
            "The margin/time combination is unusual for a late-game special-situation prompt. Double-check the score margin."
        )

    if defense_style and not isinstance(defense_style, str):
        warnings.append("Defense style was parsed in a non-text shape and may not explain cleanly downstream.")

    if context.get("need") == "two_for_one" and time_remaining is not None and time_remaining > 40:
        warnings.append(
            "2-for-1 was requested, but the parsed time remaining is longer than a typical true 2-for-1 window."
        )

    if context.get("need") == "need3" and margin is not None and margin >= 0:
        warnings.append(
            "The prompt suggests needing a 3, but the parsed margin is not negative. Verify whether the score state was interpreted correctly."
        )

    return warnings


def context_objective_sentence(context: Dict[str, Any]) -> str:
    need = context.get("need")
    margin = _to_float(context.get("margin"))
    time_remaining = _to_float(context.get("time_remaining"))
    defense_style = context.get("defense_style")

    if need == "need3":
        return "Priority is generating a clean 3-point shot without wasting clock."
    if need == "quick2":
        return "Priority is generating a fast, efficient 2-point look and staying ahead of the clock."
    if need == "two_for_one":
        return "Priority is getting the first shot up early enough to preserve the final-possession window."
    if need == "stop":
        return "Priority is the defensive possession: protect the paint, finish the stop, and avoid bailout fouls."
    if need == "safe":
        return "Priority is possession security: value the ball and avoid a dead possession."
    if need == "last_shot":
        return "Priority is controlling the last shot timing while still creating a clean look."
    if need == "foul_game":
        return "Priority is managing the foul-game decision and protecting against a tying 3."
    if margin is not None and time_remaining is not None and margin < 0 and time_remaining <= 45:
        return "Priority is urgency with control: attack quickly, but do not settle for a forced shot."
    if margin is not None and time_remaining is not None and margin > 0 and time_remaining <= 45:
        return "Priority is score-and-clock management: get a quality look without opening the door to a quick runout."
    if defense_style == "switch":
        return "Priority is creating a clean advantage against switching without letting the possession stall."
    if isinstance(defense_style, str) and defense_style.startswith("zone"):
        return "Priority is stressing the gaps, shifting the zone, and finishing before it resets."
    return "Priority is selecting the highest-efficiency option for the current game state."


def describe_context_brief(context: Dict[str, Any]) -> str:
    bits: List[str] = []

    period = context.get("period")
    if period == 5:
        bits.append("OT")
    elif isinstance(period, int) and 1 <= period <= 4:
        bits.append(f"Q{period}")

    time_remaining = _to_float(context.get("time_remaining"))
    if time_remaining is not None:
        minutes = int(time_remaining // 60)
        seconds = int(round(time_remaining % 60))
        bits.append(f"{minutes}:{seconds:02d} left")

    margin = _to_float(context.get("margin"))
    if margin is not None:
        if margin > 0:
            bits.append(f"up {int(margin) if float(margin).is_integer() else margin:g}")
        elif margin < 0:
            bits.append(f"down {abs(int(margin)) if float(margin).is_integer() else abs(margin):g}")
        else:
            bits.append("tied")

    need = need_label(context.get("need"))
    if need:
        bits.append(need)

    defense = defense_style_label(context.get("defense_style"))
    if defense:
        bits.append(defense)

    pace = pace_label(context.get("pace"))
    if pace:
        bits.append(pace)

    specials = special_situation_labels(context)
    bits.extend(specials[:2])

    return " • ".join(_dedupe_keep_order(bits))


__all__ = [
    "PLAY_FAMILY_ALIASES",
    "build_context_warning",
    "context_objective_sentence",
    "defense_style_label",
    "derive_context_flags",
    "describe_context_brief",
    "family_fit_sentences",
    "infer_play_family_from_name",
    "need_label",
    "pace_label",
    "special_situation_labels",
]