"""Timeout Quick-Call Mode: turn a ranking into three large, readable calls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

from domain.baseline_recommendation.playtype_zones import best_shot_zone

QUICK_CALL_SIZE = 3
REASON_MAX_CHARS = 110

SOURCE_CONTEXT = "context"
SOURCE_BASELINE = "baseline"


@dataclass(frozen=True)
class QuickCallPlay:
    rank: int
    play_type: str
    ppp: float
    shot_zone: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= REASON_MAX_CHARS:
        return text
    return text[: REASON_MAX_CHARS - 1].rstrip(" ,;") + "…"


def one_line_reason(row: Mapping[str, Any], source: str) -> str:
    """One short sentence a coach can read at a glance."""
    if source == SOURCE_CONTEXT:
        delta = _number(row, "DELTA_VS_BASELINE")
        # Labels look like "Late & trailing | must_score"; the first part is the game state.
        state = str(row.get("CONTEXT_LABEL") or "").split("|")[0].strip()
        if delta is not None:
            reason = f"{delta:+.2f} PPP vs season baseline"
            if state and state.lower() != "normal context":
                reason += f" ({state.lower()})"
            return _shorten(reason)
    else:
        ours = _number(row, "PPP_OFF_SHRUNK")
        theirs = _number(row, "PPP_DEF_SHRUNK")
        if ours is not None and theirs is not None:
            return _shorten(f"Our offense {ours:.2f} PPP vs their defense allowing {theirs:.2f}")
    return _shorten(str(row.get("RATIONALE") or "Highest expected points for this matchup"))


def build_quick_call(rows: Sequence[Mapping[str, Any]], source: str) -> List[QuickCallPlay]:
    """Top three plays with a shot zone and a one-line reason each."""
    ppp_key = "PPP_CONTEXT" if source == SOURCE_CONTEXT else "PPP_PRED"
    plays: List[QuickCallPlay] = []
    for row in rows:
        play_type = str(row.get("PLAY_TYPE") or "").strip()
        ppp = _number(row, ppp_key)
        if not play_type or ppp is None:
            continue
        plays.append(
            QuickCallPlay(
                rank=len(plays) + 1,
                play_type=play_type,
                ppp=round(ppp, 3),
                shot_zone=best_shot_zone(play_type),
                reason=one_line_reason(row, source),
            )
        )
        if len(plays) == QUICK_CALL_SIZE:
            break
    return plays
