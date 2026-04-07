"""NLP router: parse natural-language game context and explain recommendations."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from application.api_coordination.auth_dependency import require_auth

try:
    from infrastructure.external_integrations import (
        parse_game_context,
        context_to_context_ml_params,
        explain_recommendations,
        explain_shotplan,
    )
except Exception:  # pragma: no cover
    from nlp_parser import parse_game_context, context_to_context_ml_params  # type: ignore
    from nlp_explain import explain_recommendations, explain_shotplan  # type: ignore

router = APIRouter(prefix="/nlp", tags=["nlp"])

# Request / Response models

class ParseRequest(BaseModel):
    text: str = Field(..., description="Natural-language description of the game situation.")
    defaults: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional fallback values for missing fields (e.g., period/time_remaining/margin).",
    )

class ParseResponse(BaseModel):
    context: Dict[str, Any]
    confidence: float
    clarifying_questions: list[str]
    matches: Dict[str, str]
    # These are the exact params your /rank-plays/context-ml expects (when available)
    context_ml_params: Optional[Dict[str, Any]] = None

class ExplainRequest(BaseModel):
    # Parsed/structured context (from /nlp/parse or manual form)
    context: Dict[str, Any] = Field(..., description="Structured game context used for explanations.")

    # Ranked plays output (from /rank-plays/context-ml). We accept ANY JSON shape.
    ranked_context: Any = Field(..., description="Context-ML ranking output (list or wrapped dict).")

    # Optional baseline ranking output (from /rank-plays/baseline). Any JSON shape.
    ranked_baseline: Optional[Any] = Field(
        default=None,
        description="Baseline ranking output (optional). Used to compute simple deltas when possible.",
    )

    top_k: int = Field(default=5, ge=1, le=20, description="How many top plays to explain.")

    # Optional shotplan output (from /shotplan/rank). If provided, we explain it too.
    shotplan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional shotplan output to include in the explanation payload.",
    )

class ExplainResponse(BaseModel):
    context_summary: str
    overall_summary: str
    plays: Any
    notes: list[str]
    shotplan_explanation: Optional[Dict[str, Any]] = None

def _to_plain_dict(obj: Any) -> Any:
    """Convert Pydantic models/dataclasses to plain dicts safely."""
    # Pydantic v2: model_dump; v1: dict
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj

# Endpoints

@router.post("/parse", response_model=ParseResponse, dependencies=[Depends(require_auth)])
def nlp_parse(req: ParseRequest) -> ParseResponse:
    """Parse natural language into structured game context."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    result = parse_game_context(text, defaults=req.defaults)

    # If we have all required fields, expose ready-to-use params for your context-ML endpoint
    context_ml_params: Optional[Dict[str, Any]] = None
    try:
        context_ml_params = context_to_context_ml_params(result.context)
    except Exception:
        context_ml_params = None

    return ParseResponse(
        context=result.context,
        confidence=float(result.confidence),
        clarifying_questions=list(result.clarifying_questions),
        matches=dict(result.matches),
        context_ml_params=context_ml_params,
    )

@router.post("/explain", response_model=ExplainResponse, dependencies=[Depends(require_auth)])
def nlp_explain(req: ExplainRequest) -> ExplainResponse:
    """Build deterministic explanations for ranked plays (and optionally shotplan),"""
    if not isinstance(req.context, dict):
        raise HTTPException(status_code=422, detail="context must be an object/dict")

    # Deterministic play explanations
    try:
        exp = explain_recommendations(
            context=req.context,
            ranked_context=req.ranked_context,
            ranked_baseline=req.ranked_baseline,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {e}")

    exp_dict = _to_plain_dict(exp)

    # Optional shotplan explanation
    shotplan_expl: Optional[Dict[str, Any]] = None
    if req.shotplan is not None:
        try:
            shotplan_expl = explain_shotplan(req.context, req.shotplan)
        except Exception:
            # Keep it non-fatal; shotplan is optional
            shotplan_expl = None

    return ExplainResponse(
        context_summary=str(exp_dict.get("context_summary", "")),
        overall_summary=str(exp_dict.get("overall_summary", "")),
        plays=exp_dict.get("plays", []),
        notes=list(exp_dict.get("notes", [])),
        shotplan_explanation=shotplan_expl,
    )
