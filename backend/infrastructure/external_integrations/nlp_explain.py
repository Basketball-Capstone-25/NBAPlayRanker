from __future__ import annotations

"""
backend/nlp_explain.py

Deterministic basketball explanation builder.

Purpose:
- turn ranked play outputs + parsed NLP context into coach-friendly explanations
- support both backend-native ranking payloads and frontend-remapped payloads
- preserve advanced context signals (need3, quick2, switching, ATO, etc.)
- mention extracted basketball context more clearly
- stay metrics-backed, stable, and defendable
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence

try:
    from .nlp_reasoning import (  # type: ignore
        context_objective_sentence,
        defense_style_label,
        describe_context_brief,
        family_fit_sentences,
        infer_play_family_from_name,
        need_label,
        pace_label,
        special_situation_labels,
    )
    from .nlp_taxonomy import family_label  # type: ignore
except Exception:  # pragma: no cover
    from nlp_reasoning import (
        context_objective_sentence,
        defense_style_label,
        describe_context_brief,
        family_fit_sentences,
        infer_play_family_from_name,
        need_label,
        pace_label,
        special_situation_labels,
    )
    from nlp_taxonomy import family_label


EXPLAINER_VERSION = "3.0.0"


@dataclass(frozen=True)
class PlayExplanation:
    play_name: str
    play_type: str
    rank: int
    summary: str
    evidence: List[str] = field(default_factory=list)
    caution: Optional[str] = None
    matched_context: List[str] = field(default_factory=list)
    metrics_used: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplanationResult:
    context_summary: str
    overall_summary: str
    plays: List[PlayExplanation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    parser_warnings: List[str] = field(default_factory=list)
    clarifying_questions: List[str] = field(default_factory=list)
    explainer_version: str = EXPLAINER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_EFF_KEYS = (
    "PPP_CONTEXT",
    "PPP_PRED",
    "PPP_ML_BLEND",
    "PPP_BASELINE",
    "ppp",
    "pppPred",
    "finalPPP",
    "mlPPP",
    "baselinePPP",
    "predicted_ppp",
    "expected_ppp",
    "expected_points_per_possession",
)

_DELTA_KEYS = (
    "DELTA_VS_BASELINE",
    "delta_vs_baseline",
    "deltaPPP",
    "pppGap",
    "PPP_GAP",
    "lift",
    "improvement",
)

_CONTEXT_PPP_KEYS = ("PPP_CONTEXT", "finalPPP", "context_ppp")
_ML_PPP_KEYS = ("PPP_ML_BLEND", "mlPPP", "ml_ppp")
_BASELINE_PPP_KEYS = ("PPP_BASELINE", "baselinePPP", "PPP_PRED", "pppPred", "baseline_ppp")
_CTX_LABEL_KEYS = ("CONTEXT_LABEL", "contextLabel", "context_label")
_RATIONALE_KEYS = ("RATIONALE", "rationale")
_CTX_ADJ_KEYS = ("CONTEXT_ADJ", "context_adj", "context_adjustment")
_FREQ_KEYS = ("freq", "frequency", "usage", "share", "rate", "pct", "percent")
_COUNT_KEYS = ("n", "N", "count", "possessions", "samples", "attempts")
_TOV_KEYS = ("tov", "turnover_rate", "to_rate", "TOV_RATE", "turnovers")
_FOUL_KEYS = ("foul_rate", "ft_rate", "FTA_RATE", "ftr", "free_throw_rate")
_OFF_PPP_KEYS = ("PPP_OFF", "ppp_off", "our_ppp", "offense_ppp")
_DEF_PPP_KEYS = ("PPP_DEF", "ppp_def", "their_ppp_allowed", "defense_ppp")
_RAW_RANK_KEYS = ("rank", "RANK", "position", "idx")


def _is_number(x: Any) -> bool:
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except Exception:
        return False


def _to_float(x: Any) -> Optional[float]:
    if _is_number(x):
        return float(x)
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _fmt_clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or not _is_number(seconds):
        return None
    s = max(0, int(round(float(seconds))))
    return f"{s // 60}:{s % 60:02d}"


def _fmt_pct(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    v = float(x)
    if v > 1.0:
        v = v / 100.0
    v = _clamp(v, 0.0, 1.0)
    return f"{v * 100:.0f}%"


def _fmt_ppp(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    return f"{float(x):.2f} PPP"


def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_rankings(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key in (
            "rankings",
            "results",
            "data",
            "items",
            "plays",
            "recommendations",
            "top_k",
            "top_plays",
            "explanation",
        ):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        if obj and any(isinstance(v, (int, float, str, list, dict)) for v in obj.values()):
            return [obj]

    return []


def _play_name(play: Dict[str, Any]) -> str:
    for key in (
        "play_name",
        "play_type",
        "PLAY_TYPE",
        "playType",
        "play",
        "name",
        "action",
        "label",
        "type",
    ):
        value = play.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    meta = play.get("meta")
    if isinstance(meta, dict):
        for key in ("play_type", "name", "label"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raw = play.get("raw")
    if isinstance(raw, dict):
        return _play_name(raw)

    return "Unknown Play"


def _extract_metrics(play: Dict[str, Any]) -> Dict[str, Any]:
    raw = play.get("raw") if isinstance(play.get("raw"), dict) else None

    eff = _to_float(_first_present(play, _EFF_KEYS))
    if eff is None and raw is not None:
        eff = _to_float(_first_present(raw, _EFF_KEYS))

    context_ppp = _to_float(_first_present(play, _CONTEXT_PPP_KEYS))
    if context_ppp is None and raw is not None:
        context_ppp = _to_float(_first_present(raw, _CONTEXT_PPP_KEYS))

    ml_ppp = _to_float(_first_present(play, _ML_PPP_KEYS))
    if ml_ppp is None and raw is not None:
        ml_ppp = _to_float(_first_present(raw, _ML_PPP_KEYS))

    baseline_ppp = _to_float(_first_present(play, _BASELINE_PPP_KEYS))
    if baseline_ppp is None and raw is not None:
        baseline_ppp = _to_float(_first_present(raw, _BASELINE_PPP_KEYS))

    delta = _to_float(_first_present(play, _DELTA_KEYS))
    if delta is None and raw is not None:
        delta = _to_float(_first_present(raw, _DELTA_KEYS))

    context_adj = _to_float(_first_present(play, _CTX_ADJ_KEYS))
    if context_adj is None and raw is not None:
        context_adj = _to_float(_first_present(raw, _CTX_ADJ_KEYS))

    freq = _to_float(_first_present(play, _FREQ_KEYS))
    if freq is None and raw is not None:
        freq = _to_float(_first_present(raw, _FREQ_KEYS))

    count_raw = _first_present(play, _COUNT_KEYS)
    if count_raw is None and raw is not None:
        count_raw = _first_present(raw, _COUNT_KEYS)
    count = int(float(count_raw)) if _is_number(count_raw) else None

    tov = _to_float(_first_present(play, _TOV_KEYS))
    if tov is None and raw is not None:
        tov = _to_float(_first_present(raw, _TOV_KEYS))

    foul = _to_float(_first_present(play, _FOUL_KEYS))
    if foul is None and raw is not None:
        foul = _to_float(_first_present(raw, _FOUL_KEYS))

    off_ppp = _to_float(_first_present(play, _OFF_PPP_KEYS))
    if off_ppp is None and raw is not None:
        off_ppp = _to_float(_first_present(raw, _OFF_PPP_KEYS))

    def_ppp = _to_float(_first_present(play, _DEF_PPP_KEYS))
    if def_ppp is None and raw is not None:
        def_ppp = _to_float(_first_present(raw, _DEF_PPP_KEYS))

    ctx_label = _first_present(play, _CTX_LABEL_KEYS)
    if ctx_label is None and raw is not None:
        ctx_label = _first_present(raw, _CTX_LABEL_KEYS)

    rationale = _first_present(play, _RATIONALE_KEYS)
    if rationale is None and raw is not None:
        rationale = _first_present(raw, _RATIONALE_KEYS)

    top_factors = play.get("top_factors") or play.get("feature_contrib") or play.get("shap_top")
    if top_factors is None and raw is not None:
        top_factors = raw.get("top_factors") or raw.get("feature_contrib") or raw.get("shap_top")

    metrics: Dict[str, Any] = {}

    if eff is not None:
        metrics["ppp"] = eff
    if context_ppp is not None:
        metrics["context_ppp"] = context_ppp
    if ml_ppp is not None:
        metrics["ml_ppp"] = ml_ppp
    if baseline_ppp is not None:
        metrics["baseline_ppp"] = baseline_ppp
    if delta is not None:
        metrics["delta_vs_baseline"] = delta
    if context_adj is not None:
        metrics["context_adj"] = context_adj
    if freq is not None:
        metrics["freq"] = freq
    if count is not None:
        metrics["count"] = count
    if tov is not None:
        metrics["turnover_rate"] = tov
    if foul is not None:
        metrics["foul_rate"] = foul
    if off_ppp is not None:
        metrics["off_ppp"] = off_ppp
    if def_ppp is not None:
        metrics["def_ppp"] = def_ppp
    if isinstance(ctx_label, str) and ctx_label.strip():
        metrics["context_label"] = ctx_label.strip()
    if isinstance(rationale, str) and rationale.strip():
        metrics["rationale"] = rationale.strip()
    if top_factors is not None:
        metrics["top_factors"] = top_factors

    return metrics


def summarize_context(context: Dict[str, Any]) -> str:
    if isinstance(context.get("context_brief"), str) and context["context_brief"].strip():
        return context["context_brief"].strip()

    brief = describe_context_brief(context)
    if brief:
        return brief

    parts: List[str] = []

    period = context.get("period")
    if isinstance(period, int):
        parts.append("OT" if period == 5 else f"Q{period}")

    clock = _fmt_clock(_to_float(context.get("time_remaining")))
    if clock:
        parts.append(clock)

    shot_clock = _to_float(context.get("shot_clock"))
    if shot_clock is not None:
        parts.append(f"{int(round(shot_clock))} on shot clock")

    margin = _to_float(context.get("margin"))
    if margin is not None:
        if abs(margin) < 0.001:
            parts.append("tied")
        elif margin < 0:
            parts.append(f"down {abs(margin):g}")
        else:
            parts.append(f"up {margin:g}")

    need = need_label(context.get("need"))
    if need:
        parts.append(need)

    defense = defense_style_label(context.get("defense_style"))
    if defense:
        parts.append(f"vs {defense}")

    pace = pace_label(context.get("pace"))
    if pace:
        parts.append(pace)

    parts.extend(special_situation_labels(context))

    return " • ".join(_dedupe_keep_order(parts)) if parts else "Game context"


def _format_top_factors(tf: Any, max_items: int = 3) -> Optional[str]:
    items: List[str] = []

    if isinstance(tf, list):
        for entry in tf:
            if len(items) >= max_items:
                break
            if isinstance(entry, str) and entry.strip():
                items.append(entry.strip())
            elif isinstance(entry, dict):
                feat = entry.get("feature") or entry.get("name")
                if isinstance(feat, str) and feat.strip():
                    items.append(feat.strip())
            elif isinstance(entry, (tuple, list)) and entry:
                feat = entry[0]
                if isinstance(feat, str) and feat.strip():
                    items.append(feat.strip())

    return ", ".join(items) if items else None


def _format_freeform_rationale(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    return cleaned.rstrip(".") + "."


def _context_fit_tags(context: Dict[str, Any], play_name: str) -> List[str]:
    tags: List[str] = []

    play_family = infer_play_family_from_name(play_name)
    requested_families = list(context.get("preferred_play_families") or [])
    if play_family and play_family in requested_families:
        tags.append(family_label(play_family))

    if context.get("after_timeout"):
        tags.append("after-timeout")
    if context.get("slob"):
        tags.append("SLOB")
    if context.get("blob"):
        tags.append("BLOB")
    if context.get("advance_ball"):
        tags.append("advanced-ball")
    if context.get("vs_switching"):
        tags.append("switch coverage")
    if context.get("need3"):
        tags.append("need-3 urgency")
    if context.get("quick2"):
        tags.append("quick-2 urgency")
    if context.get("two_for_one"):
        tags.append("2-for-1")
    if context.get("hold_for_last"):
        tags.append("last-shot")
    if context.get("must_stop"):
        tags.append("stop need")
    if context.get("protect_lead"):
        tags.append("protect lead")
    if context.get("late_clock"):
        tags.append("late clock")

    return _dedupe_keep_order(tags)


def _choose_caution(
    context: Dict[str, Any],
    play_metrics: Dict[str, Any],
    play_name: str,
) -> Optional[str]:
    need = context.get("need")
    defense_style = context.get("defense_style")
    play_family = infer_play_family_from_name(play_name)
    turnover_rate = _to_float(play_metrics.get("turnover_rate"))
    foul_rate = _to_float(play_metrics.get("foul_rate"))

    if turnover_rate is not None and turnover_rate >= 0.18:
        return "Caution: turnover risk is relatively high here, so spacing and entry timing matter."
    if context.get("two_for_one"):
        return "Caution: keep the first action quick enough to preserve the final-possession window."
    if need == "need3":
        return "Caution: do not settle for a heavily contested 3 just because the clock is low."
    if need == "quick2":
        return "Caution: do not burn extra time searching for a perfect look—take the first clean advantage."
    if need == "foul_game":
        return "Caution: the score state points to foul-game management, so clock and free-throw math matter as much as shot quality."
    if context.get("protect_lead") and turnover_rate is not None and turnover_rate >= 0.14:
        return "Caution: protecting the lead raises the cost of a live-ball turnover here."
    if defense_style == "switch" and play_family == "ball_screen":
        return "Caution: if they switch cleanly, have the slip or mismatch counter ready immediately."
    if defense_style == "drop" and play_family == "ball_screen":
        return "Caution: if the pull-up is not there, get off it early rather than dribbling into the drop."
    if isinstance(defense_style, str) and defense_style.startswith("zone"):
        return "Caution: versus zone, weakside timing and gap occupation matter more than holding the ball."
    if foul_rate is not None and foul_rate < 0.05 and need == "quick2":
        return "Caution: this option does not project much foul pressure, so finishing cleanly matters."

    return None


def _build_extracted_context_notes(context: Dict[str, Any]) -> List[str]:
    notes: List[str] = []

    period = context.get("period")
    time_remaining = _to_float(context.get("time_remaining"))
    shot_clock = _to_float(context.get("shot_clock"))
    margin = _to_float(context.get("margin"))
    need = context.get("need")
    needs = list(context.get("needs") or [])
    defense_style = context.get("defense_style")
    pace = context.get("pace")
    special_situations = list(context.get("special_situations") or [])
    play_families = list(context.get("preferred_play_families") or [])
    parser_version = context.get("parser_version")
    pipeline_version = context.get("nlp_pipeline_version")

    core_parts: List[str] = []
    if isinstance(period, int):
        core_parts.append("OT" if period == 5 else f"Q{period}")
    if time_remaining is not None:
        core_parts.append(f"{_fmt_clock(time_remaining)} remaining")
    if shot_clock is not None:
        core_parts.append(f"{int(round(shot_clock))} on the shot clock")
    if margin is not None:
        if abs(margin) < 0.001:
            core_parts.append("game tied")
        elif margin < 0:
            core_parts.append(f"down {abs(margin):g}")
        else:
            core_parts.append(f"up {margin:g}")
    if core_parts:
        notes.append("Extracted game state: " + ", ".join(core_parts) + ".")

    if need or needs:
        label = need_label(need) if need else None
        extra_needs = [need_label(x) or str(x) for x in needs if x != need]
        if label and extra_needs:
            notes.append("Extracted objective: " + label + " with secondary signals of " + ", ".join(extra_needs) + ".")
        elif label:
            notes.append("Extracted objective: " + label + ".")
        elif extra_needs:
            notes.append("Extracted objective signals: " + ", ".join(extra_needs) + ".")

    if defense_style:
        defense_text = defense_style_label(defense_style) or str(defense_style)
        notes.append(f"Extracted defense context: {defense_text}.")

    if pace:
        pace_text = pace_label(pace) or str(pace)
        notes.append(f"Extracted pace intent: {pace_text}.")

    if special_situations:
        situation_labels = special_situation_labels(context)
        if situation_labels:
            notes.append("Extracted special situations: " + ", ".join(situation_labels) + ".")

    if play_families:
        labels = [family_label(x) for x in play_families]
        notes.append("Extracted play-family cues: " + ", ".join(_dedupe_keep_order(labels)) + ".")

    if pipeline_version:
        if parser_version:
            notes.append(f"Explanation used parser {parser_version} with NLP pipeline {pipeline_version}.")
        else:
            notes.append(f"Explanation used NLP pipeline {pipeline_version}.")

    return _dedupe_keep_order(notes)


def _evidence_bullets(context: Dict[str, Any], play_name: str, metrics: Dict[str, Any]) -> List[str]:
    bullets: List[str] = []

    context_ppp = _to_float(metrics.get("context_ppp"))
    ppp = _to_float(metrics.get("ppp")) or context_ppp
    ml_ppp = _to_float(metrics.get("ml_ppp"))
    baseline_ppp = _to_float(metrics.get("baseline_ppp"))
    delta = _to_float(metrics.get("delta_vs_baseline"))
    context_adj = _to_float(metrics.get("context_adj"))
    off_ppp = _to_float(metrics.get("off_ppp"))
    def_ppp = _to_float(metrics.get("def_ppp"))

    if context_ppp is not None:
        bullets.append(f"Context-adjusted efficiency: {_fmt_ppp(context_ppp)}.")
    elif ppp is not None:
        bullets.append(f"Efficiency signal: {_fmt_ppp(ppp)}.")

    if ml_ppp is not None and baseline_ppp is not None:
        bullets.append(f"Blend components: ML {_fmt_ppp(ml_ppp)} vs baseline {_fmt_ppp(baseline_ppp)}.")
    elif baseline_ppp is not None and delta is None:
        bullets.append(f"Baseline matchup signal: {_fmt_ppp(baseline_ppp)}.")

    if delta is not None:
        sign = "+" if delta >= 0 else ""
        bullets.append(f"Lift vs baseline: {sign}{delta:.2f} PPP.")
    elif context_adj is not None:
        sign = "+" if context_adj >= 0 else ""
        bullets.append(f"Context adjustment: {sign}{context_adj:.2f}.")

    if off_ppp is not None and def_ppp is not None:
        bullets.append(f"Matchup profile: our side {_fmt_ppp(off_ppp)} vs their allowance {_fmt_ppp(def_ppp)}.")

    freq = _to_float(metrics.get("freq"))
    if freq is not None:
        pct = _fmt_pct(freq)
        if pct:
            bullets.append(f"Usage signal: {pct} in the underlying sample.")

    count = metrics.get("count")
    if isinstance(count, int) and count > 0:
        bullets.append(f"Sample size: {count} tracked possessions/entries.")

    fit_lines = family_fit_sentences(list(context.get("preferred_play_families") or []), play_name)
    bullets.extend(fit_lines)

    ctx_label = metrics.get("context_label")
    if isinstance(ctx_label, str) and ctx_label.strip():
        bullets.append(f"Model context label: {ctx_label.strip()}.")

    rationale = _format_freeform_rationale(metrics.get("rationale"))
    if rationale:
        bullets.append(f"Model rationale: {rationale}")

    top_factors = _format_top_factors(metrics.get("top_factors"))
    if top_factors:
        bullets.append(f"Top drivers: {top_factors}.")

    return bullets[:6]


def explain_play(play: Dict[str, Any], context: Dict[str, Any], rank: int) -> PlayExplanation:
    name = _play_name(play)
    metrics = _extract_metrics(play)
    play_family = infer_play_family_from_name(name)

    context_ppp = _to_float(metrics.get("context_ppp"))
    ppp = context_ppp or _to_float(metrics.get("ppp"))
    delta = _to_float(metrics.get("delta_vs_baseline"))
    need = need_label(context.get("need"))
    defense = defense_style_label(context.get("defense_style"))
    requested_families = list(context.get("preferred_play_families") or [])

    if ppp is not None:
        summary = f"{name} ranks #{rank} here at {_fmt_ppp(ppp)}."
    else:
        summary = f"{name} ranks #{rank} here based on the available matchup and context signals."

    if play_family and requested_families and play_family in requested_families:
        summary = f"{name} ranks #{rank} here and aligns with the prompt’s {family_label(play_family)} cue."

    if delta is not None and ppp is not None and not (play_family and requested_families and play_family in requested_families):
        sign = "+" if delta >= 0 else ""
        summary = f"{name} ranks #{rank} here at {_fmt_ppp(ppp)}, with {sign}{delta:.2f} PPP versus baseline."

    if need and defense:
        summary += f" It fits the parsed {need.lower()} situation against {defense.lower()}."
    elif need:
        summary += f" It fits the parsed {need.lower()} situation."
    elif defense:
        summary += f" It is being evaluated against the parsed {defense.lower()} look."

    evidence = _evidence_bullets(context, name, metrics)
    caution = _choose_caution(context, metrics, name)
    matched_context = _context_fit_tags(context, name)

    return PlayExplanation(
        play_name=name,
        play_type=name,
        rank=rank,
        summary=summary,
        evidence=evidence,
        caution=caution,
        matched_context=matched_context,
        metrics_used=metrics,
    )


def explain_recommendations(
    context: Dict[str, Any],
    ranked_context: Any,
    ranked_baseline: Any = None,
    top_k: int = 5,
    parser_warnings: Optional[List[str]] = None,
    clarifying_questions: Optional[List[str]] = None,
) -> ExplanationResult:
    context_summary = summarize_context(context)
    overall_summary = context.get("objective_summary") or context_objective_sentence(context)

    context_list = _coerce_rankings(ranked_context)
    baseline_list = _coerce_rankings(ranked_baseline) if ranked_baseline is not None else []
    baseline_by_name: Dict[str, Dict[str, Any]] = {_play_name(b): b for b in baseline_list}

    plays_out: List[PlayExplanation] = []
    notes: List[str] = []
    notes.extend(_build_extracted_context_notes(context))

    limit = max(1, int(top_k))

    for i, play in enumerate(context_list[:limit], start=1):
        rank_raw = _first_present(play, _RAW_RANK_KEYS)
        rank = int(float(rank_raw)) if _is_number(rank_raw) else i

        explained = explain_play(play, context, rank=rank)

        baseline_play = baseline_by_name.get(explained.play_name)
        if baseline_play is not None:
            baseline_metrics = _extract_metrics(baseline_play)
            baseline_ppp = _to_float(baseline_metrics.get("context_ppp")) or _to_float(baseline_metrics.get("ppp"))
            current_ppp = _to_float(explained.metrics_used.get("context_ppp")) or _to_float(explained.metrics_used.get("ppp"))

            if (
                baseline_ppp is not None
                and current_ppp is not None
                and "delta_vs_baseline" not in explained.metrics_used
            ):
                delta_val = current_ppp - baseline_ppp
                sign = "+" if delta_val >= 0 else ""
                explained.evidence.append(f"Context vs baseline: {sign}{delta_val:.2f} PPP for this situation.")

        plays_out.append(
            PlayExplanation(
                play_name=explained.play_name,
                play_type=explained.play_type,
                rank=explained.rank,
                summary=explained.summary,
                evidence=_dedupe_keep_order(explained.evidence),
                caution=explained.caution,
                matched_context=_dedupe_keep_order(explained.matched_context),
                metrics_used=explained.metrics_used,
            )
        )

    missing_core = [k for k in ("period", "time_remaining", "margin") if context.get(k) is None]
    if missing_core:
        notes.append(
            f"Some core context fields are missing ({', '.join(missing_core)}), so the explanation is less specific than it could be."
        )

    requested_families = list(context.get("preferred_play_families") or [])
    if requested_families and plays_out:
        if not any(infer_play_family_from_name(p.play_name) in requested_families for p in plays_out):
            labels = ", ".join(family_label(f) for f in requested_families)
            notes.append(
                f"The prompt suggested {labels}, but none of the top returned play families matched exactly. That can happen if the ranking model prefers a different family in this matchup."
            )

    if context.get("need3") and plays_out:
        if all("3" not in p.play_name.lower() for p in plays_out[: min(3, len(plays_out))]):
            notes.append(
                "The parsed game state suggests a 3 may be required, but the top recommendations are not obviously 3-point-oriented by name. Double-check score-clock intent versus the model’s efficiency preference."
            )

    if context.get("quick2") and plays_out:
        top_name = plays_out[0].play_name.lower()
        if "post" in top_name or "iso" in top_name:
            notes.append(
                "The parsed game state suggests quick-2 urgency. If the top option becomes too slow to enter, move immediately to the next clean trigger."
            )

    if context.get("two_for_one"):
        notes.append("2-for-1 logic increases the value of shot timing, not only raw expected PPP.")

    if context.get("must_stop"):
        notes.append("The parsed prompt includes a stop need, so offensive rankings alone may not fully answer the possession-management question.")

    if context.get("after_timeout"):
        notes.append("Because the parsed prompt includes an after-timeout situation, set quality and immediate execution matter more than generic half-court value.")

    if context.get("slob") or context.get("blob"):
        inbound_context = "sideline" if context.get("slob") else "baseline"
        notes.append(
            f"The parsed prompt includes a {inbound_context} out-of-bounds situation, so space, timing, and first-option clarity matter more than normal flow offense."
        )

    if not context_list:
        notes.append("No ranked context plays were provided, so there was nothing to explain.")

    merged_parser_warnings = _dedupe_keep_order(
        list(parser_warnings or []) + list(context.get("warnings") or [])
    )
    merged_clarifying_questions = _dedupe_keep_order(list(clarifying_questions or []))

    return ExplanationResult(
        context_summary=context_summary,
        overall_summary=str(overall_summary),
        plays=plays_out,
        notes=_dedupe_keep_order(notes),
        parser_warnings=merged_parser_warnings,
        clarifying_questions=merged_clarifying_questions,
    )


def explain_shotplan(context: Dict[str, Any], shotplan: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(shotplan, dict):
        return out

    shot_type = shotplan.get("shot_type") or shotplan.get("shotType") or shotplan.get("type")
    zone = shotplan.get("zone") or shotplan.get("shot_zone") or shotplan.get("shotZone")
    shooter = shotplan.get("shooter") or shotplan.get("player") or shotplan.get("name")

    if not shot_type and isinstance(shotplan.get("top_shot_types"), list) and shotplan["top_shot_types"]:
        first = shotplan["top_shot_types"][0]
        if isinstance(first, dict):
            shot_type = first.get("SHOT_TYPE") or first.get("shot_type") or first.get("type")

    if not zone and isinstance(shotplan.get("top_zones"), list) and shotplan["top_zones"]:
        first = shotplan["top_zones"][0]
        if isinstance(first, dict):
            zone = first.get("ZONE") or first.get("zone")

    parts: List[str] = []
    if isinstance(shot_type, str) and shot_type:
        parts.append(shot_type)
    if isinstance(zone, str) and zone:
        parts.append(zone)
    if isinstance(shooter, str) and shooter:
        parts.append(f"via {shooter}")

    if parts:
        out["summary"] = "Best shot-plan direction: " + " • ".join(parts)

    expected_ppp = _to_float(
        shotplan.get("expected_ppp")
        or shotplan.get("expected_value")
        or shotplan.get("expected_points")
        or shotplan.get("ppp")
    )

    evidence: List[str] = []
    if expected_ppp is not None:
        evidence.append(f"Expected efficiency: {_fmt_ppp(expected_ppp)}.")

    if isinstance(shotplan.get("rationale"), str) and shotplan["rationale"].strip():
        formatted = _format_freeform_rationale(shotplan["rationale"])
        if formatted:
            evidence.append(formatted)

    period = context.get("period")
    time_remaining = _to_float(context.get("time_remaining"))
    margin = _to_float(context.get("margin"))
    if period is not None or time_remaining is not None or margin is not None:
        context_bits: List[str] = []
        if isinstance(period, int):
            context_bits.append("OT" if period == 5 else f"Q{period}")
        if time_remaining is not None:
            context_bits.append(_fmt_clock(time_remaining) or "")
        if margin is not None:
            if abs(margin) < 0.001:
                context_bits.append("tied")
            elif margin < 0:
                context_bits.append(f"down {abs(margin):g}")
            else:
                context_bits.append(f"up {margin:g}")
        context_bits = [x for x in context_bits if x]
        if context_bits:
            evidence.append("Parsed context: " + " • ".join(context_bits) + ".")

    out["evidence"] = [x for x in evidence if x]

    if context.get("need3") and isinstance(shot_type, str) and "3" not in shot_type:
        out["caution"] = (
            "Caution: the parsed game state suggests a 3 may be required, so double-check whether this shot type matches the score-clock need."
        )
    else:
        out["caution"] = None

    return out


if __name__ == "__main__":
    context_demo = {
        "period": 4,
        "time_remaining": 28,
        "margin": -3,
        "need": "quick2",
        "needs": ["quick2", "must_score"],
        "defense_style": "switch",
        "pace": "push",
        "preferred_play_families": ["ball_screen"],
        "special_situations": ["after_timeout"],
        "after_timeout": True,
        "vs_switching": True,
        "quick2": True,
        "context_brief": "Q4 • 0:28 left • down 3 • quick 2 • switching • after timeout",
        "objective_summary": "Priority is generating a fast, efficient 2-point look and staying ahead of the clock.",
        "parser_version": "3.0.0",
        "nlp_pipeline_version": "1.0.0",
    }

    ranked_context_demo = {
        "rankings": [
            {
                "play_type": "P&R Ball Handler",
                "PPP_CONTEXT": 1.08,
                "PPP_ML_BLEND": 1.06,
                "PPP_BASELINE": 1.00,
                "DELTA_VS_BASELINE": 0.08,
            },
            {
                "play_type": "Post Up",
                "PPP_CONTEXT": 1.02,
                "PPP_BASELINE": 1.00,
            },
        ]
    }

    print(explain_recommendations(context_demo, ranked_context_demo).to_dict())