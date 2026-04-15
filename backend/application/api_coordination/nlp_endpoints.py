from __future__ import annotations

"""
backend/nlp_endpoints.py

FastAPI router for basketball NLP features.

Endpoints:
- GET  /nlp/health
- POST /nlp/parse
- POST /nlp/explain

Design goals:
- preserve backward compatibility with the current Gameplan payloads
- expose richer parser signals for future frontend + ML upgrades
- never let minor payload shape differences crash the router
- preserve the same main /nlp/parse and /nlp/explain API shape
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from .nlp_parser import (  # type: ignore
        NLPParseResult,
        context_to_context_ml_params,
        parse_game_context,
    )
    from .nlp_explain import (  # type: ignore
        ExplanationResult,
        explain_recommendations,
        explain_shotplan,
    )
except Exception:  # pragma: no cover
    from nlp_parser import NLPParseResult, context_to_context_ml_params, parse_game_context
    from nlp_explain import ExplanationResult, explain_recommendations, explain_shotplan


router = APIRouter(prefix="/nlp", tags=["nlp"])


class ParseRequest(BaseModel):
    text: str = Field(..., description="Natural-language basketball game situation.")
    defaults: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional fallback UI values such as period, time_remaining, margin, and shot_clock.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("text is required")
        if len(cleaned) > 2000:
            raise ValueError("text is too long")
        return cleaned


class ParseResponse(BaseModel):
    context: Dict[str, Any]
    confidence: float
    clarifying_questions: List[str] = Field(default_factory=list)
    matches: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    context_ml_params: Optional[Dict[str, Any]] = None
    parser_version: Optional[str] = None
    raw_text: Optional[str] = None


class ExplainRequest(BaseModel):
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured parsed context. Preferred when already available.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Optional raw coaching prompt. If supplied, the router can parse and merge it into context.",
    )
    defaults: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional fallback values used only when parsing text inside /nlp/explain.",
    )

    ranked_context: Optional[Any] = Field(
        default=None,
        description="Context-aware ranking payload. Preferred field name.",
    )
    ranked_baseline: Optional[Any] = Field(
        default=None,
        description="Baseline ranking payload. Optional but useful for delta explanations.",
    )
    top_k: int = Field(default=5, ge=1, le=20)

    shotplan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional shot-plan payload to explain alongside play-type recommendations.",
    )

    parser_warnings: Optional[List[str]] = Field(
        default=None,
        description="Optional warnings previously returned by /nlp/parse.",
    )
    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description="Optional questions previously returned by /nlp/parse.",
    )

    # Legacy frontend compatibility
    mode: Optional[str] = Field(default=None, description="Legacy mode field.")
    rankings: Optional[Any] = Field(default=None, description="Legacy rankings field.")
    top_n: Optional[int] = Field(default=None, description="Legacy count field.")

    @model_validator(mode="after")
    def validate_payload(self) -> "ExplainRequest":
        has_context = isinstance(self.context, dict) and bool(self.context)
        has_text = bool((self.text or "").strip())
        has_rankings = self.ranked_context is not None or self.rankings is not None
        has_shotplan = isinstance(self.shotplan, dict) and bool(self.shotplan)

        if not has_context and not has_text:
            raise ValueError("Provide either context or text.")

        if not has_rankings and not has_shotplan:
            raise ValueError(
                "Provide ranked_context (or legacy rankings), or provide shotplan for shot-plan explanation."
            )

        return self


class ExplainResponse(BaseModel):
    context_summary: str
    overall_summary: str
    plays: Any
    notes: List[str] = Field(default_factory=list)
    parser_warnings: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    shotplan_explanation: Optional[Dict[str, Any]] = None
    explainer_version: Optional[str] = None

    # Legacy frontend compatibility
    mode: Optional[str] = None
    explanation: Any = None


class HealthResponse(BaseModel):
    status: str
    router: str
    ready: bool


@router.get("/health", response_model=HealthResponse)
def nlp_health() -> HealthResponse:
    return HealthResponse(status="ok", router="nlp", ready=True)


def _to_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
    return out


def _coerce_context(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="context must be an object/dict")
    return dict(value)


def _coerce_top_k(top_k: int, top_n: Optional[int]) -> int:
    resolved = top_k
    if top_n is not None:
        try:
            resolved = int(top_n)
        except Exception:
            resolved = top_k
    return max(1, min(20, int(resolved)))


def _resolve_rank_payloads(req: ExplainRequest) -> Dict[str, Any]:
    ranked_context = req.ranked_context
    ranked_baseline = req.ranked_baseline
    mode = (req.mode or "").strip().lower()

    if ranked_context is None and req.rankings is not None:
        ranked_context = req.rankings
        if mode == "baseline" and ranked_baseline is None:
            ranked_baseline = req.rankings

    return {
        "ranked_context": ranked_context,
        "ranked_baseline": ranked_baseline,
    }


def _merge_context(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _safe_parse_text(text: str, defaults: Optional[Dict[str, Any]]) -> NLPParseResult:
    try:
        return parse_game_context(text=text, defaults=defaults)
    except TypeError:
        try:
            return parse_game_context(text, defaults=defaults)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse text: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse text: {exc}")


def _build_overall_summary(context: Dict[str, Any]) -> str:
    objective_summary = context.get("objective_summary")
    if isinstance(objective_summary, str) and objective_summary.strip():
        return objective_summary.strip()

    context_brief = context.get("context_brief")
    if isinstance(context_brief, str) and context_brief.strip():
        return f"Parsed context: {context_brief.strip()}"

    return "Parsed basketball context for downstream recommendation and explanation."


@router.post("/parse", response_model=ParseResponse)
def nlp_parse(req: ParseRequest) -> ParseResponse:
    result = _safe_parse_text(req.text, req.defaults)

    context_ml_params: Optional[Dict[str, Any]] = None
    try:
        context_ml_params = context_to_context_ml_params(result.context)
    except Exception:
        context_ml_params = None

    context = dict(result.context)
    if "warnings" not in context:
        context["warnings"] = list(result.warnings)

    parser_version = context.get("parser_version")
    parser_version_str = str(parser_version) if parser_version is not None else None

    return ParseResponse(
        context=context,
        confidence=float(result.confidence),
        clarifying_questions=list(result.clarifying_questions),
        matches=dict(result.matches),
        warnings=list(result.warnings),
        context_ml_params=context_ml_params,
        parser_version=parser_version_str,
        raw_text=str(result.raw_text or req.text),
    )


@router.post("/explain", response_model=ExplainResponse)
def nlp_explain(req: ExplainRequest) -> ExplainResponse:
    resolved_context = _coerce_context(req.context)

    parse_result: Optional[NLPParseResult] = None
    if (req.text or "").strip():
        parse_result = _safe_parse_text(req.text.strip(), req.defaults)
        resolved_context = _merge_context(parse_result.context, resolved_context)

    resolved_rankings = _resolve_rank_payloads(req)
    resolved_top_k = _coerce_top_k(req.top_k, req.top_n)

    parser_warnings: List[str] = []
    clarifying_questions: List[str] = []

    if parse_result is not None:
        parser_warnings.extend(list(parse_result.warnings))
        clarifying_questions.extend(list(parse_result.clarifying_questions))

    if isinstance(req.parser_warnings, list):
        parser_warnings.extend([str(x) for x in req.parser_warnings if isinstance(x, str)])

    if isinstance(req.clarifying_questions, list):
        clarifying_questions.extend([str(x) for x in req.clarifying_questions if isinstance(x, str)])

    if isinstance(resolved_context.get("warnings"), list):
        parser_warnings.extend(
            [str(x) for x in resolved_context.get("warnings", []) if isinstance(x, str)]
        )

    parser_warnings = _dedupe_keep_order(parser_warnings)
    clarifying_questions = _dedupe_keep_order(clarifying_questions)

    exp_dict: Dict[str, Any]
    if resolved_rankings["ranked_context"] is not None:
        try:
            exp: ExplanationResult = explain_recommendations(
                context=resolved_context,
                ranked_context=resolved_rankings["ranked_context"],
                ranked_baseline=resolved_rankings["ranked_baseline"],
                top_k=resolved_top_k,
                parser_warnings=parser_warnings,
                clarifying_questions=clarifying_questions,
            )
        except TypeError:
            try:
                exp = explain_recommendations(
                    context=resolved_context,
                    ranked_context=resolved_rankings["ranked_context"],
                    ranked_baseline=resolved_rankings["ranked_baseline"],
                    top_k=resolved_top_k,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {exc}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {exc}")

        exp_dict = _to_plain_dict(exp)
    else:
        exp_dict = {
            "context_summary": (
                str(resolved_context.get("context_brief"))
                if isinstance(resolved_context.get("context_brief"), str)
                else "Game context"
            ),
            "overall_summary": _build_overall_summary(resolved_context),
            "plays": [],
            "notes": [],
            "parser_warnings": parser_warnings,
            "clarifying_questions": clarifying_questions,
            "explainer_version": None,
        }

    shotplan_explanation: Optional[Dict[str, Any]] = None
    if req.shotplan is not None:
        try:
            shotplan_explanation = explain_shotplan(resolved_context, req.shotplan)
        except Exception:
            shotplan_explanation = None

    plays = exp_dict.get("plays", [])
    legacy_mode = (req.mode or "context-ml").strip().lower() or "context-ml"

    return ExplainResponse(
        context_summary=str(exp_dict.get("context_summary", "")),
        overall_summary=str(exp_dict.get("overall_summary", "")),
        plays=plays,
        notes=list(exp_dict.get("notes", [])),
        parser_warnings=list(exp_dict.get("parser_warnings", parser_warnings)),
        clarifying_questions=list(exp_dict.get("clarifying_questions", clarifying_questions)),
        shotplan_explanation=shotplan_explanation,
        explainer_version=(
            str(exp_dict.get("explainer_version"))
            if exp_dict.get("explainer_version") is not None
            else None
        ),
        mode=legacy_mode,
        explanation=plays,
    )