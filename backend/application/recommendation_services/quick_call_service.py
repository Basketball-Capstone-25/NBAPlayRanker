"""Quick-Call service: context-aware top 3, with a labelled baseline fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import pandas as pd

from domain.context_ml_recommendation.quick_call import (
    QUICK_CALL_SIZE,
    SOURCE_BASELINE,
    SOURCE_CONTEXT,
    QuickCallPlay,
    build_quick_call,
)

logger = logging.getLogger(__name__)

BASELINE_NOTICE = "Game context not applied: the context model is unavailable, so these are baseline plays."


@dataclass(frozen=True)
class QuickCallRequest:
    season: str
    our_team: str
    opp_team: str
    margin: float
    period: int
    time_remaining: float


@dataclass
class QuickCallResult:
    source: str
    notice: str
    plays: List[QuickCallPlay] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "notice": self.notice,
            "plays": [play.to_dict() for play in self.plays],
        }


ContextRanker = Callable[[QuickCallRequest], pd.DataFrame]
BaselineRanker = Callable[[QuickCallRequest], pd.DataFrame]


class QuickCallService:
    """Ask the context model first; if it fails or returns nothing, use the baseline."""

    def __init__(self, context_ranker: ContextRanker, baseline_ranker: BaselineRanker) -> None:
        self._context_ranker = context_ranker
        self._baseline_ranker = baseline_ranker

    def recommend(self, request: QuickCallRequest) -> QuickCallResult:
        try:
            rows = self._context_ranker(request).to_dict(orient="records")
            plays = build_quick_call(rows, SOURCE_CONTEXT)
            if plays:
                return QuickCallResult(SOURCE_CONTEXT, "", plays)
            logger.warning("quick-call: context model returned no usable plays; using baseline")
        except Exception as exc:  # any model failure must still give the coach a call
            logger.warning("quick-call: context model unavailable (%s); using baseline", type(exc).__name__)

        rows = self._baseline_ranker(request).to_dict(orient="records")
        return QuickCallResult(SOURCE_BASELINE, BASELINE_NOTICE, build_quick_call(rows, SOURCE_BASELINE))


__all__ = [
    "BASELINE_NOTICE",
    "QUICK_CALL_SIZE",
    "QuickCallRequest",
    "QuickCallResult",
    "QuickCallService",
]
