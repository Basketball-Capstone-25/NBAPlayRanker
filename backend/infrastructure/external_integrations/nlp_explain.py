"""Explanation generator for recommendation results."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Data structures

@dataclass(frozen=True)
class PlayExplanation:
    play_name: str
    summary: str
    evidence: List[str] = field(default_factory=list)
    caution: Optional[str] = None
    metrics_used: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExplanationResult:
    context_summary: str
    overall_summary: str
    plays: List[PlayExplanation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses convert nested dataclasses fine, but we ensure JSON-friendly types
        return d

# Helpers

def _is_number(x: Any) -> bool:
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except Exception:
        return False

def _to_float(x: Any) -> Optional[float]:
    return float(x) if _is_number(x) else None

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _fmt_clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or not _is_number(seconds):
        return None
    s = int(round(float(seconds)))
    s = max(0, s)
    mm = s // 60
    ss = s % 60
    return f"{mm}:{ss:02d}"

def _fmt_pct(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    # If it's already likely a percent (e.g., 35), handle it.
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
    """Accepts a list of dicts OR a response dict with common wrappers:"""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("rankings", "results", "data", "items", "plays"):
            v = obj.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # Sometimes a single play dict is returned
        if obj and any(isinstance(v, (int, float, str)) for v in obj.values()):
            return [obj]
    return []

def _play_name(play: Dict[str, Any]) -> str:
    """Try several common keys for a human-readable play label."""
    for k in ("play_type", "playType", "play", "name", "action", "label", "type"):
        v = play.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: try nested shape
    meta = play.get("meta")
    if isinstance(meta, dict):
        v = meta.get("play_type") or meta.get("name")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Unknown Play"

# Metric extraction (flexible key mapping)

# Efficiency / expected points per possession
_EFF_KEYS = (
    "PPP_PRED",
    "ppp_pred",
    "ppp",
    "predicted_ppp",
    "expected_ppp",
    "expected_points_per_possession",
)

# Delta/lift vs baseline (optional)
_DELTA_KEYS = (
    "DELTA_VS_BASELINE",
    "delta_vs_baseline",
    "lift",
    "improvement",
)

# Context adjustment (optional)
_CTX_ADJ_KEYS = (
    "CONTEXT_ADJ",
    "context_adj",
    "context_adjustment",
)

# Frequency / usage (optional)
_FREQ_KEYS = (
    "freq",
    "frequency",
    "usage",
    "share",
    "rate",
    "pct",
    "percent",
)

# Sample size / count (optional)
_COUNT_KEYS = (
    "n",
    "N",
    "count",
    "possessions",
    "samples",
    "attempts",
)

# Risk metrics (optional)
_TOV_KEYS = ("tov", "turnover_rate", "to_rate", "TOV_RATE", "turnovers")
_FOUL_KEYS = ("foul_rate", "ft_rate", "FTA_RATE", "ftr", "free_throw_rate")

def _extract_metrics(play: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a normalized set of key metrics if present."""
    eff = _to_float(_first_present(play, _EFF_KEYS))
    delta = _to_float(_first_present(play, _DELTA_KEYS))
    ctx_adj = _to_float(_first_present(play, _CTX_ADJ_KEYS))
    freq_raw = _first_present(play, _FREQ_KEYS)
    freq = _to_float(freq_raw) if freq_raw is not None else None
    count_raw = _first_present(play, _COUNT_KEYS)
    count = int(count_raw) if isinstance(count_raw, int) else (int(float(count_raw)) if _is_number(count_raw) else None)

    tov = _to_float(_first_present(play, _TOV_KEYS))
    foul = _to_float(_first_present(play, _FOUL_KEYS))

    out: Dict[str, Any] = {}
    if eff is not None:
        out["ppp"] = eff
    if delta is not None:
        out["delta_vs_baseline"] = delta
    if ctx_adj is not None:
        out["context_adj"] = ctx_adj
    if freq is not None:
        out["freq"] = freq
    if count is not None:
        out["count"] = count
    if tov is not None:
        out["turnover_rate"] = tov
    if foul is not None:
        out["foul_rate"] = foul

    # Optional: include top factors if present in common shapes
    # e.g. play["top_factors"] = [{"feature":"...", "direction":"+","value":...}, ...]
    # or play["top_factors"] = [("feature", 0.12), ...]
    tf = play.get("top_factors") or play.get("feature_contrib") or play.get("shap_top")
    if tf is not None:
        out["top_factors"] = tf

    return out

# Context summarization

def summarize_context(context: Dict[str, Any]) -> str:
    """Build a short, human-readable summary of the parsed context."""
    parts: List[str] = []

    period = context.get("period")
    if isinstance(period, int):
        if period == 5:
            parts.append("OT")
        else:
            parts.append(f"Q{period}")

    clock = _fmt_clock(_to_float(context.get("time_remaining")))
    if clock:
        parts.append(clock)

    margin = _to_float(context.get("margin"))
    if margin is not None:
        if abs(margin) < 0.001:
            parts.append("tied")
        elif margin < 0:
            parts.append(f"down {int(abs(margin)) if float(margin).is_integer() else abs(margin):g}")
        else:
            parts.append(f"up {int(margin) if float(margin).is_integer() else margin:g}")

    need = context.get("need")
    if isinstance(need, str) and need:
        # friendly label
        mapping = {
            "quick2": "quick 2",
            "need3": "need a 3",
            "stop": "need a stop",
            "two_for_one": "2-for-1",
            "safe": "protect possession",
        }
        parts.append(mapping.get(need, need))

    defense_style = context.get("defense_style")
    if isinstance(defense_style, str) and defense_style:
        mapping = {
            "switch": "vs switching",
            "drop": "vs drop",
            "zone_2_3": "vs 2-3 zone",
            "zone_3_2": "vs 3-2 zone",
            "zone_1_3_1": "vs 1-3-1",
            "box_and_1": "vs box-and-1",
        }
        parts.append(mapping.get(defense_style, f"vs {defense_style}"))

    pace = context.get("pace")
    if isinstance(pace, str) and pace:
        parts.append("push pace" if pace == "push" else ("slow pace" if pace == "slow" else pace))

    return " • ".join(parts) if parts else "Game context"

# Explanation generation

def _context_intent_sentence(context: Dict[str, Any]) -> str:
    """A single sentence describing the objective implied by context."""
    need = context.get("need")
    if need == "need3":
        return "Priority is generating a clean 3-point look quickly."
    if need == "quick2":
        return "Priority is creating a quick, high-quality 2-point look."
    if need == "two_for_one":
        return "Priority is optimizing for a 2-for-1 sequence (quick shot + last possession)."
    if need == "stop":
        return "Priority is securing a stop and avoiding transition breakdowns."
    if need == "safe":
        return "Priority is protecting the ball and avoiding empty possessions."
    # If margin/time suggests urgency, we can hint without inventing.
    margin = _to_float(context.get("margin"))
    tr = _to_float(context.get("time_remaining"))
    if margin is not None and tr is not None and tr <= 45 and margin < 0:
        return "Priority is getting a quality shot quickly while managing the clock."
    return "Recommendations are selected to maximize expected efficiency for the given situation."

def _choose_caution(context: Dict[str, Any], play_metrics: Dict[str, Any]) -> Optional[str]:
    """Deterministic caution based on known signals."""
    need = context.get("need")
    defense_style = context.get("defense_style")

    # If we know turnover rate, caution on high turnover
    tov = _to_float(play_metrics.get("turnover_rate"))
    if tov is not None and tov >= 0.18:
        return "Caution: turnover risk is relatively high—prioritize clean spacing and secure entry."

    if need == "two_for_one":
        return "Caution: keep the first action quick enough to preserve the final possession window."

    if defense_style == "switch":
        return "Caution: expect switches—have a slip/short-roll counter ready if the first option is contained."

    if need == "need3":
        return "Caution: don’t force a contested 3—use one extra advantage-creating pass if it’s not clean."

    if need == "quick2":
        return "Caution: don’t burn time hunting a perfect look—take the best clean advantage early."

    return None

def _evidence_bullets(context: Dict[str, Any], m: Dict[str, Any]) -> List[str]:
    """Build up to ~3 evidence bullets using only available metrics."""
    bullets: List[str] = []

    ppp = _to_float(m.get("ppp"))
    if ppp is not None:
        bullets.append(f"Efficiency: {_fmt_ppp(ppp)} (model estimate).")

    delta = _to_float(m.get("delta_vs_baseline"))
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        bullets.append(f"Lift vs baseline: {sign}{delta:.2f} PPP.")

    ctx_adj = _to_float(m.get("context_adj"))
    if ctx_adj is not None:
        sign = "+" if ctx_adj >= 0 else ""
        bullets.append(f"Context adjustment: {sign}{ctx_adj:.2f} (directional boost for this situation).")

    freq = _to_float(m.get("freq"))
    if freq is not None:
        pct = _fmt_pct(freq)
        if pct:
            bullets.append(f"Usage signal: {pct} (how often this action appears in the dataset).")

    count = m.get("count")
    if isinstance(count, int) and count > 0:
        bullets.append(f"Sample size: {count} possessions/entries (where available).")

    tov = _to_float(m.get("turnover_rate"))
    if tov is not None:
        pct = _fmt_pct(tov)
        if pct:
            bullets.append(f"Turnover rate: {pct} (lower is safer).")

    foul = _to_float(m.get("foul_rate"))
    if foul is not None:
        pct = _fmt_pct(foul)
        if pct:
            bullets.append(f"Foul/FT signal: {pct} (chance of drawing fouls/free throws).")

    # Optional: top factors (if present)
    # We keep this extremely cautious: we only mention labels, not invented magnitudes.
    tf = m.get("top_factors")
    if tf:
        top_feats = _format_top_factors(tf)
        if top_feats:
            bullets.append(f"Top drivers: {top_feats}.")

    # Keep at most 3-4 bullets; UI-friendly
    return bullets[:4]

def _format_top_factors(tf: Any, max_items: int = 3) -> Optional[str]:
    """Attempt to format top explanatory factors from common shapes."""
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
            elif isinstance(entry, (tuple, list)) and len(entry) >= 1:
                feat = entry[0]
                if isinstance(feat, str) and feat.strip():
                    items.append(feat.strip())

    if not items:
        return None
    return ", ".join(items)

def explain_play(
    play: Dict[str, Any],
    context: Dict[str, Any],
) -> PlayExplanation:
    """Build a deterministic explanation for one play."""
    name = _play_name(play)
    metrics = _extract_metrics(play)

    # Summary: prefer efficiency sentence if we have PPP
    ppp = _to_float(metrics.get("ppp"))
    if ppp is not None:
        summary = f"{name} rates well here at {_fmt_ppp(ppp)}."
    else:
        summary = f"{name} is recommended for this context based on available model signals."

    evidence = _evidence_bullets(context, metrics)
    caution = _choose_caution(context, metrics)

    return PlayExplanation(
        play_name=name,
        summary=summary,
        evidence=evidence,
        caution=caution,
        metrics_used=metrics,
    )

def explain_recommendations(
    context: Dict[str, Any],
    ranked_context: Any,
    ranked_baseline: Any = None,
    top_k: int = 5,
) -> ExplanationResult:
    """Build an overall explanation package from ranked play outputs."""
    ctx_summary = summarize_context(context)
    overall = _context_intent_sentence(context)

    ctx_list = _coerce_rankings(ranked_context)
    base_list = _coerce_rankings(ranked_baseline) if ranked_baseline is not None else []

    # Build a quick lookup for baseline by play name (best-effort)
    baseline_by_name: Dict[str, Dict[str, Any]] = {}
    for b in base_list:
        baseline_by_name[_play_name(b)] = b

    plays_out: List[PlayExplanation] = []
    notes: List[str] = []

    for play in ctx_list[: max(1, int(top_k))]:
        pe = explain_play(play, context)

        # Optional: if baseline PPP exists and context PPP exists, compute delta note
        b = baseline_by_name.get(pe.play_name)
        if b is not None:
            b_metrics = _extract_metrics(b)
            b_ppp = _to_float(b_metrics.get("ppp"))
            c_ppp = _to_float(pe.metrics_used.get("ppp"))
            if b_ppp is not None and c_ppp is not None:
                delta = c_ppp - b_ppp
                # Add a small extra evidence line only if we can compute it honestly
                sign = "+" if delta >= 0 else ""
                extra = f"Context vs baseline: {sign}{delta:.2f} PPP for this situation."
                # Avoid duplication if delta already provided by model
                if "delta_vs_baseline" not in pe.metrics_used:
                    pe.evidence.append(extra)

        plays_out.append(pe)

    # Add a gentle note if context missing core fields
    missing = []
    for k in ("period", "time_remaining", "margin"):
        if context.get(k) is None:
            missing.append(k)
    if missing:
        notes.append(f"Some context fields are missing ({', '.join(missing)}), so explanations may be less specific.")

    return ExplanationResult(
        context_summary=ctx_summary,
        overall_summary=overall,
        plays=plays_out,
        notes=notes,
    )

# Optional shot-plan explanation (if you want it)

def explain_shotplan(context: Dict[str, Any], shotplan: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic explanation for shot-plan output."""
    out: Dict[str, Any] = {}
    if not isinstance(shotplan, dict):
        return out

    shot_type = shotplan.get("shot_type") or shotplan.get("shotType") or shotplan.get("type")
    zone = shotplan.get("zone") or shotplan.get("shot_zone") or shotplan.get("shotZone")
    shooter = shotplan.get("shooter") or shotplan.get("player") or shotplan.get("name")

    parts: List[str] = []
    if isinstance(shot_type, str) and shot_type:
        parts.append(shot_type)
    if isinstance(zone, str) and zone:
        parts.append(zone)
    if isinstance(shooter, str) and shooter:
        parts.append(f"via {shooter}")

    if parts:
        out["summary"] = "Best shot option: " + " • ".join(parts)

    eff = _to_float(_first_present(shotplan, _EFF_KEYS + ("expected_value", "expected_points")))
    if eff is not None:
        out["evidence"] = [f"Expected efficiency: {_fmt_ppp(eff)} (from shot-plan model/aggregate)."]
    else:
        out["evidence"] = []

    # context-aware caution
    need = context.get("need")
    if need == "need3" and isinstance(shot_type, str) and "3" not in shot_type:
        out["caution"] = "Caution: your intent indicates a 3 may be required—consider a 3-point option if needed."
    else:
        out["caution"] = None

    return out

# Quick manual demo
