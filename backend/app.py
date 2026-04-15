from __future__ import annotations

"""
backend/app.py

Basketball Strategy backend (FastAPI)

This version keeps the existing endpoints working while upgrading the
context-ML flow so Gameplan/NLP can send richer basketball context.

Key NLP/Gameplan upgrades in this file:
- keeps the existing GET /rank-plays/context-ml endpoint for backwards compatibility
- adds POST /rank-plays/context-ml for richer NLP payloads
- accepts advanced parsed context fields from /nlp/parse
- pushes all ranking logic through backend/ml_context_recommender.py
  so NLP affects the ranking itself, not only explanation text
- keeps router loading fault-tolerant so NLP/PBP imports cannot break the whole API
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------
# IMPORTANT: make backend/ imports work no matter where uvicorn is run from
# (e.g., repo root: uvicorn backend.app:app --reload)
# ---------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger("basketball_strategy")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from baseline_recommender import BaselineRecommender, rank_playtypes_baseline
from export_pdf import create_pdf_router
from ml_context_recommender import (
    rank_ml_with_context,
    recommender_health,
    sanitize_context_for_ranking,
    validate_context_guardrails,
)
from ml_models import paired_t_test_rmse, run_cv_evaluation
from ml_stat_analysis import compute_ml_analysis
from shot_baseline_recommender import ShotBaselineRecommender
from shot_etl import CLEAN_PARQUET
from shot_ml_models import run_shot_model_cv
from shot_ml_stat_analysis import compute_shot_ml_analysis

# IMPORTANT:
# Do NOT import viz_sportypy here at module load time.
# If SportyPy isn't installed (or is slow/import-heavy), it can prevent the API from starting.
# We'll lazy-import it inside the /viz endpoint.

# ---------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------

app = FastAPI(
    title="Basketball Strategy API",
    description=(
        "Backend for the Basketball Strategy Analysis capstone.\n\n"
        "Endpoints map directly to the frontend pages:\n"
        "- /meta/options: dropdown options\n"
        "- /data/team-playtypes: raw aggregated dataset preview + filtering\n"
        "- /rank-plays/baseline: transparent baseline recommender\n"
        "- /rank-plays/context-ml: AI use case (ML + context)\n"
        "- /metrics/baseline-vs-ml: holdout evaluation (defense evidence)\n"
    ),
)

# ---------------------------------------------------------------------
# IMPORTANT: Dataset2 router should NOT be able to break Dataset1 startup.
# If /pbp modules have an import error, we still want play type pages working.
# ---------------------------------------------------------------------
try:
    from pbp_endpoints import router as pbp_router  # type: ignore

    app.include_router(pbp_router)
    logger.info("Loaded Dataset2 (/pbp) router successfully.")
except Exception as e:
    logger.warning(
        "Dataset2 (/pbp) router NOT loaded due to import error: %s. "
        "Dataset1 playtype endpoints will still work.",
        e,
    )

# ---------------------------------------------------------------------
# OPTIONAL shots explorer router.
# ---------------------------------------------------------------------
try:
    from pbp_shots_endpoints import router as pbp_shots_router  # type: ignore

    app.include_router(pbp_shots_router)
    logger.info("Loaded Dataset2 (/pbp) shots explorer endpoints successfully.")
except Exception as e:
    logger.warning(
        "Dataset2 (/pbp) shots explorer router NOT loaded (this is ok if you haven't created it yet): %s. "
        "Core endpoints will still work.",
        e,
    )

# ---------------------------------------------------------------------
# IMPORTANT: NLP router should NOT be able to break startup either.
# ---------------------------------------------------------------------
try:
    from nlp_endpoints import router as nlp_router  # type: ignore

    app.include_router(nlp_router)
    logger.info("Loaded NLP (/nlp) router successfully.")
except Exception as e:
    logger.warning(
        "NLP (/nlp) router NOT loaded due to import error: %s. "
        "Core endpoints will still work.",
        e,
    )

# Allow local dev + keep permissive for defense demo environments.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Startup cache (important for multi-user + performance)
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SYNERGY_CSV = DATA_DIR / "synergy_playtypes_2019_2025_players.csv"
ML_PRED_CSV = DATA_DIR / "ml_offense_ppp_predictions.csv"

# Load baseline tables ONCE and reuse (fast for multi-user requests).
rec = BaselineRecommender(str(SYNERGY_CSV))

# Cache ML predictions ONCE (if file exists).
ML_PRED_DF: Optional[pd.DataFrame] = None
if ML_PRED_CSV.exists():
    ML_PRED_DF = pd.read_csv(ML_PRED_CSV)

# Shot Intelligence caches (lazy-loaded)
SHOT_REC: Optional[ShotBaselineRecommender] = None
SHOT_CLEAN_DF: Optional[pd.DataFrame] = None

# Precompute meta options from the dataset so we don’t hardcode team/season lists.
VALID_SEASONS = sorted(rec.team_df["SEASON"].dropna().unique().tolist())
VALID_TEAMS = sorted(rec.team_df["TEAM_ABBREVIATION"].dropna().unique().tolist())
VALID_PLAYTYPES = sorted(rec.team_df["PLAY_TYPE"].dropna().unique().tolist())
VALID_SIDES = ["offense", "defense"]

TEAM_NAMES: Dict[str, str] = {}
try:
    tmp = rec.team_df[["TEAM_ABBREVIATION", "TEAM_NAME"]].dropna().drop_duplicates()
    TEAM_NAMES = {r["TEAM_ABBREVIATION"]: r["TEAM_NAME"] for _, r in tmp.iterrows()}
except Exception:
    TEAM_NAMES = {}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _require_season(season: str) -> None:
    if season not in VALID_SEASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown season '{season}'. Allowed: {VALID_SEASONS}",
        )


def _require_team(team: str, label: str) -> None:
    if team not in VALID_TEAMS:
        raise HTTPException(status_code=400, detail=f"Unknown {label} team code '{team}'.")


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records (NaN -> None)."""
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _get_shot_rec() -> ShotBaselineRecommender:
    global SHOT_REC
    if SHOT_REC is None:
        try:
            SHOT_REC = ShotBaselineRecommender()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Shot aggregates not found. Run:\n"
                    "  python backend/shot_aggregates.py\n"
                    f"Missing: {e}"
                ),
            )
    return SHOT_REC


def _get_shots_clean_df() -> pd.DataFrame:
    global SHOT_CLEAN_DF
    if SHOT_CLEAN_DF is None:
        if not CLEAN_PARQUET.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    "shots_clean.parquet not found. Run:\n"
                    "  python backend/data/etl/build_shots_dataset.py"
                ),
            )
        SHOT_CLEAN_DF = pd.read_parquet(CLEAN_PARQUET)
    return SHOT_CLEAN_DF


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    except Exception:
        return None


def _safe_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        value = x.strip()
        return value if value else None
    return None


def _as_list_str(x: Any) -> List[str]:
    out: List[str] = []
    seen = set()

    if isinstance(x, list):
        items = x
    elif isinstance(x, tuple):
        items = list(x)
    else:
        return out

    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)

    return out


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


# ---------------------------------------------------------------------
# NLP-aware context request model + normalization helpers
# ---------------------------------------------------------------------


class ContextMLRequest(BaseModel):
    season: str
    our: str
    opp: str

    # Legacy flat fields
    margin: Optional[float] = None
    period: Optional[int] = Field(default=None, ge=1, le=5)
    time_remaining: Optional[float] = Field(default=None, ge=0, le=720)
    k: int = Field(default=5, ge=1, le=10)
    w_off: float = Field(default=0.7, ge=0, le=1)

    shot_clock: Optional[float] = Field(default=None, ge=0, le=24)
    need: Optional[str] = None
    needs: List[str] = Field(default_factory=list)
    defense_style: Optional[str] = None
    pace: Optional[str] = None

    after_timeout: bool = False
    slob: bool = False
    blob: bool = False
    advance_ball: bool = False
    late_clock: bool = False
    need3: bool = False
    protect_lead: bool = False
    end_of_quarter: bool = False
    vs_switching: bool = False
    must_stop: bool = False
    quick2: bool = False
    two_for_one: bool = False
    hold_for_last: bool = False
    foul_game: bool = False
    no_three: bool = False
    must_score: bool = False
    safe: bool = False

    special_situations: List[str] = Field(default_factory=list)
    preferred_play_families: List[str] = Field(default_factory=list)
    intent_tags: List[str] = Field(default_factory=list)
    offense_bias: Optional[float] = Field(default=None, ge=0, le=1)
    defense_bias: Optional[float] = Field(default=None, ge=0, le=1)

    # New richer nested NLP payload from Gameplan / /nlp/parse
    context: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def merge_nested_context(self) -> "ContextMLRequest":
        ctx = self.context if isinstance(self.context, dict) else {}
        if not ctx:
            return self

        if self.margin is None:
            parsed = _safe_float(ctx.get("margin"))
            if parsed is not None:
                self.margin = parsed

        if self.period is None:
            parsed_period = _safe_float(ctx.get("period"))
            if parsed_period is not None:
                rounded = int(round(parsed_period))
                if 1 <= rounded <= 5:
                    self.period = rounded

        if self.time_remaining is None:
            parsed_time = _safe_float(ctx.get("time_remaining"))
            if parsed_time is not None:
                self.time_remaining = parsed_time

        if self.shot_clock is None:
            parsed_shot_clock = _safe_float(ctx.get("shot_clock"))
            if parsed_shot_clock is not None:
                self.shot_clock = parsed_shot_clock

        if self.need is None:
            self.need = _safe_str(ctx.get("need"))

        if not self.needs:
            self.needs = _as_list_str(ctx.get("needs"))

        if self.defense_style is None:
            self.defense_style = _safe_str(ctx.get("defense_style"))

        if self.pace is None:
            self.pace = _safe_str(ctx.get("pace"))

        if not self.special_situations:
            self.special_situations = _as_list_str(ctx.get("special_situations"))

        if not self.preferred_play_families:
            self.preferred_play_families = _as_list_str(ctx.get("preferred_play_families"))

        if not self.intent_tags:
            self.intent_tags = _as_list_str(ctx.get("intent_tags"))

        if self.offense_bias is None:
            parsed_off = _safe_float(ctx.get("offense_bias"))
            if parsed_off is not None:
                self.offense_bias = parsed_off

        if self.defense_bias is None:
            parsed_def = _safe_float(ctx.get("defense_bias"))
            if parsed_def is not None:
                self.defense_bias = parsed_def

        bool_keys = [
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
        ]
        for key in bool_keys:
            current_value = bool(getattr(self, key))
            nested_value = _truthy(ctx.get(key))
            setattr(self, key, current_value or nested_value)

        return self


def _merge_request_context(req: ContextMLRequest) -> Dict[str, Any]:
    nested = dict(req.context or {})

    if req.margin is not None:
        nested["margin"] = req.margin
    if req.period is not None:
        nested["period"] = req.period
    if req.time_remaining is not None:
        nested["time_remaining"] = req.time_remaining
    if req.shot_clock is not None:
        nested["shot_clock"] = req.shot_clock

    if req.need:
        nested["need"] = req.need

    merged_needs = _dedupe_keep_order(
        [
            *_as_list_str(nested.get("needs")),
            *(req.needs or []),
            *([req.need] if req.need else []),
        ]
    )
    if merged_needs:
        nested["needs"] = merged_needs

    if req.defense_style:
        nested["defense_style"] = req.defense_style
    if req.pace:
        nested["pace"] = req.pace

    merged_special = _dedupe_keep_order(
        [
            *_as_list_str(nested.get("special_situations")),
            *(req.special_situations or []),
            *(["after_timeout"] if req.after_timeout else []),
            *(["slob"] if req.slob else []),
            *(["blob"] if req.blob else []),
            *(["advance_ball"] if req.advance_ball else []),
        ]
    )
    if merged_special:
        nested["special_situations"] = merged_special

    merged_families = _dedupe_keep_order(
        [
            *_as_list_str(nested.get("preferred_play_families")),
            *(req.preferred_play_families or []),
        ]
    )
    if merged_families:
        nested["preferred_play_families"] = merged_families

    merged_tags = _dedupe_keep_order(
        [
            *_as_list_str(nested.get("intent_tags")),
            *(req.intent_tags or []),
        ]
    )
    if merged_tags:
        nested["intent_tags"] = merged_tags

    if req.offense_bias is not None:
        nested["offense_bias"] = req.offense_bias
    if req.defense_bias is not None:
        nested["defense_bias"] = req.defense_bias

    bool_keys = [
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
    ]
    for key in bool_keys:
        nested[key] = bool(getattr(req, key)) or _truthy(nested.get(key))

    return nested


def _serialize_context_request(req: ContextMLRequest, applied_context: Dict[str, Any]) -> Dict[str, Any]:
    payload = req.model_dump()
    payload["w_def"] = float(1.0 - float(req.w_off))
    payload["applied_context"] = applied_context
    return payload


def _rank_context_ml_response(req: ContextMLRequest) -> Dict[str, Any]:
    _require_season(req.season)
    _require_team(req.our, "our")
    _require_team(req.opp, "opponent")
    if req.our == req.opp:
        raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

    if ML_PRED_DF is None or ML_PRED_DF.empty:
        raise HTTPException(
            status_code=400,
            detail="ML predictions not found. Run backend/ml_models.py to generate data/ml_offense_ppp_predictions.csv",
        )

    merged_context = _merge_request_context(req)
    applied_context = sanitize_context_for_ranking(
        merged_context,
        margin=req.margin,
        period=req.period,
        time_remaining=req.time_remaining,
    )

    try:
        ranked_df = rank_ml_with_context(
            season=req.season,
            our_team=req.our,
            opp_team=req.opp,
            margin=applied_context["margin"],
            period=applied_context["period"],
            time_remaining_period_sec=applied_context["time_remaining"],
            k=req.k,
            w_off=float(req.w_off),
            w_def=float(1.0 - float(req.w_off)),
            context=applied_context,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context-ML ranking failed: {e}")

    out_cols = [
        "PLAY_TYPE",
        "PLAY_FAMILY",
        "PPP_CONTEXT",
        "PPP_ML_BLEND",
        "PPP_BASELINE",
        "DELTA_VS_BASELINE",
        "CONTEXT_LABEL",
        "RATIONALE",
        "CONTEXT_ADJ",
        "NLP_CONTEXT_ADJ",
        "LEGACY_CONTEXT_ADJ",
        "NEED_ADJ",
        "DEFENSE_STYLE_ADJ",
        "PACE_ADJ",
        "SPECIAL_ADJ",
        "FAMILY_PREF_ADJ",
        "LATE_CLOCK_ADJ",
        "BALL_SECURITY_PENALTY",
        "EFFECTIVE_W_OFF",
        "EFFECTIVE_W_DEF",
        "LATE_GAME_FACTOR",
        "TRAILING_FACTOR",
        "LEADING_FACTOR",
        "CONTEXT_DEFAULTS_USED",
        "CONTEXT_GUARDRAILS",
        "CONTEXT_PARSE_STATUS",
        "CONTEXT_VALIDATION_OK",
        "RATIONALE_MODE",
    ]
    out_cols = [c for c in out_cols if c in ranked_df.columns]
    rankings = _df_to_records(ranked_df[out_cols])

    payload = _serialize_context_request(req, applied_context)

    return jsonable_encoder(
        {
            "season": req.season,
            "our_team": req.our,
            "opp_team": req.opp,
            "k": int(req.k),
            "margin": float(applied_context["margin"]),
            "period": int(applied_context["period"]),
            "time_remaining_period_sec": float(applied_context["time_remaining"]),
            "w_off": float(req.w_off),
            "w_def": float(1.0 - float(req.w_off)),
            "context_payload": payload,
            "applied_context": applied_context,
            "rankings": rankings,
        }
    )


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/context-ml")
def context_ml_health() -> Dict[str, Any]:
    return jsonable_encoder(
        {
            "status": "ok",
            "recommender": recommender_health(),
            "guardrail_validation": validate_context_guardrails(),
        }
    )


# ---------------------------------------------------------------------
# Meta endpoints (used by frontend dropdowns + defense explanations)
# ---------------------------------------------------------------------


@app.get("/meta/options")
def meta_options() -> Dict[str, Any]:
    """
    Dropdown options for the frontend.
    Derived from the dataset (no hardcoding), so it stays consistent.
    """
    return {
        "seasons": VALID_SEASONS,
        "teams": VALID_TEAMS,
        "teamNames": TEAM_NAMES,
        "playTypes": VALID_PLAYTYPES,
        "sides": VALID_SIDES,
        "hasMlPredictions": bool(ML_PRED_DF is not None and not ML_PRED_DF.empty),
    }


@app.get("/meta/pipeline")
def pipeline_info() -> Dict[str, Any]:
    return {
        "dataSource": "Synergy play-type snapshot (player rows) aggregated into team-level play-type tables (offense/defense).",
        "cleaning_and_aggregation": [
            "Map Synergy TYPE_GROUPING to SIDE = offense/defense.",
            "Group player rows into team-level rows by (SEASON, TEAM, PLAY_TYPE, SIDE).",
            "Compute possession-weighted averages for efficiency stats (PPP, eFG%, TOV%, etc.).",
            "Compute RELIABILITY_WEIGHT from log1p(POSS) to reduce noise from small samples (used for shrinkage).",
            "Build league baselines per (SEASON, PLAY_TYPE, SIDE) for shrinkage anchors.",
        ],
        "modeling": [
            "Baseline model: shrink team offense/defense toward league baselines; combine into PPP_PRED.",
            "ML model: RandomForest predicts offense PPP using team-level play-type features (offline CV).",
            "AI use case: ML-based PPP blended with opponent defense, then adjusted using score/time context.",
            "Advanced NLP context: parse-derived fields (need, defense style, shot clock, ATO, preferred play families) now flow into backend/ml_context_recommender.py so they affect ranking, not only explanation text.",
        ],
        "etl_reference": "See backend/data/etl/build_synergy_dataset.R for the dataset build logic (if applicable in your repo).",
    }


@app.get("/meta/baseline-formula")
def baseline_formula() -> Dict[str, Any]:
    return {
        "inputs": ["PPP_OFF (team offense)", "PPP_DEF (opponent defense allowed)", "league baselines", "reliability weights"],
        "shrinkage": "PPP_SHRUNK = REL * PPP_TEAM + (1-REL) * PPP_LEAGUE",
        "prediction": "PPP_PRED = w_off * PPP_OFF_SHRUNK + w_def * (2*PPP_LEAGUE_OFF - PPP_DEF_SHRUNK)",
        "defaults": {"w_off": 0.7, "w_def": 0.3},
        "interpretation": "We combine how efficient we are at a play type with how friendly the opponent is at allowing it, while stabilizing small samples using league averages.",
    }


# ---------------------------------------------------------------------
# Data Explorer endpoints (raw preview + CSV export)
# ---------------------------------------------------------------------


@app.get("/data/team-playtypes")
def team_playtypes(
    season: str = Query(..., description="Season label (required)."),
    team: Optional[str] = Query(None, description="Team abbreviation filter (optional)."),
    side: Optional[str] = Query(None, description="Side filter: offense/defense (optional)."),
    play_type: Optional[str] = Query(None, description="Play type filter (optional)."),
    min_poss: float = Query(0, ge=0, description="Minimum possessions (optional)."),
    limit: int = Query(200, ge=1, le=2000, description="Rows to return (preview limit)."),
) -> Dict[str, Any]:
    _require_season(season)
    df = rec.team_df.copy()

    df = df[df["SEASON"] == season]
    if team:
        _require_team(team, "team")
        df = df[df["TEAM_ABBREVIATION"] == team]
    if side:
        if side not in VALID_SIDES:
            raise HTTPException(status_code=400, detail="side must be 'offense' or 'defense'")
        df = df[df["SIDE"] == side]
    if play_type:
        df = df[df["PLAY_TYPE"] == play_type]
    if min_poss > 0:
        df = df[df["POSS"] >= float(min_poss)]

    total = int(df.shape[0])

    keep_cols = [
        "SEASON",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "SIDE",
        "PLAY_TYPE",
        "GP",
        "POSS",
        "POSS_PCT",
        "PPP",
        "EFG_PCT",
        "SCORE_POSS_PCT",
        "TOV_POSS_PCT",
        "RELIABILITY_WEIGHT",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].sort_values(["TEAM_ABBREVIATION", "SIDE", "PLAY_TYPE"]).head(limit)
    records = _df_to_records(df)

    return jsonable_encoder(
        {
            "season": season,
            "total_rows": total,
            "returned_rows": len(records),
            "rows": records,
        }
    )


@app.get("/data/team-playtypes.csv")
def team_playtypes_csv(
    season: str = Query(...),
    team: Optional[str] = Query(None),
    side: Optional[str] = Query(None),
    play_type: Optional[str] = Query(None),
    min_poss: float = Query(0, ge=0),
) -> StreamingResponse:
    _require_season(season)
    df = rec.team_df.copy()
    df = df[df["SEASON"] == season]

    if team:
        _require_team(team, "team")
        df = df[df["TEAM_ABBREVIATION"] == team]
    if side:
        if side not in VALID_SIDES:
            raise HTTPException(status_code=400, detail="side must be 'offense' or 'defense'")
        df = df[df["SIDE"] == side]
    if play_type:
        df = df[df["PLAY_TYPE"] == play_type]
    if min_poss > 0:
        df = df[df["POSS"] >= float(min_poss)]

    df = df.replace({np.nan: None})

    import io

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"team_playtypes_{season}.csv"
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------
# Baseline endpoint (transparent + explainable)
# ---------------------------------------------------------------------


@app.get("/rank-plays/baseline")
def rank_baseline(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation."),
    opp: str = Query(..., description="Opponent team abbreviation."),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    _require_season(season)
    _require_team(our, "our")
    _require_team(opp, "opponent")
    if our == opp:
        raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

    w_def = float(1.0 - w_off)

    try:
        df = rank_playtypes_baseline(
            team_df=rec.team_df,
            league_df=rec.league_df,
            season=season,
            our_team=our,
            opp_team=opp,
            k=k,
            w_off=float(w_off),
            w_def=w_def,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return jsonable_encoder(
        {
            "season": season,
            "our_team": our,
            "opp_team": opp,
            "k": k,
            "w_off": float(w_off),
            "w_def": float(w_def),
            "rankings": _df_to_records(df),
        }
    )


@app.get("/rank-plays/baseline.csv")
def rank_baseline_csv(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> StreamingResponse:
    w_def = float(1.0 - w_off)
    try:
        df = rank_playtypes_baseline(
            team_df=rec.team_df,
            league_df=rec.league_df,
            season=season,
            our_team=our,
            opp_team=opp,
            k=k,
            w_off=float(w_off),
            w_def=w_def,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import io

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"baseline_{season}_{our}_vs_{opp}_top{k}.csv"
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------
# AI endpoint: ML + context (GET kept for backwards compatibility)
# ---------------------------------------------------------------------


@app.get("/rank-plays/context-ml")
def rank_context_ml(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    margin: Optional[float] = Query(None, description="Our score minus opponent score."),
    period: Optional[int] = Query(None, ge=1, le=5),
    time_remaining: Optional[float] = Query(None, ge=0, le=720),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
    shot_clock: Optional[float] = Query(None, ge=0, le=24),
    need: Optional[str] = Query(None),
    defense_style: Optional[str] = Query(None),
    pace: Optional[str] = Query(None),
) -> Dict[str, Any]:
    req = ContextMLRequest(
        season=season,
        our=our,
        opp=opp,
        margin=margin,
        period=period,
        time_remaining=time_remaining,
        k=k,
        w_off=w_off,
        shot_clock=shot_clock,
        need=need,
        defense_style=defense_style,
        pace=pace,
    )
    return _rank_context_ml_response(req)


@app.post("/rank-plays/context-ml")
def rank_context_ml_post(req: ContextMLRequest) -> Dict[str, Any]:
    return _rank_context_ml_response(req)


# ---------------------------------------------------------------------
# Model evaluation endpoint (defense evidence)
# ---------------------------------------------------------------------


@app.get("/metrics/baseline-vs-ml")
def baseline_vs_ml(
    n_splits: int = Query(5, ge=2, le=10, description="K-fold splits used for evaluation."),
) -> Dict[str, Any]:
    summary_df, fold_metrics = run_cv_evaluation(n_splits=int(n_splits))

    metrics: List[Dict[str, Any]] = []
    for model_name, row in summary_df.iterrows():
        metrics.append(
            {
                "model": model_name,
                "RMSE_mean": float(row["RMSE_mean"]),
                "RMSE_std": float(row["RMSE_std"]),
                "MAE_mean": float(row["MAE_mean"]),
                "MAE_std": float(row["MAE_std"]),
                "R2_mean": float(row["R2_mean"]),
                "R2_std": float(row["R2_std"]),
            }
        )

    t_stat, p_val = paired_t_test_rmse(fold_metrics)

    return jsonable_encoder(
        {
            "n_splits": int(n_splits),
            "metrics": metrics,
            "rf_vs_baseline_t": None if (t_stat is None or np.isnan(t_stat)) else float(t_stat),
            "rf_vs_baseline_p": None if (p_val is None or np.isnan(p_val)) else float(p_val),
        }
    )


@app.get("/metrics/shot-models")
def shot_models_metrics(
    n_splits: int = Query(5, ge=2, le=10, description="GroupKFold splits by GAME_ID."),
) -> Dict[str, Any]:
    summary_df, _ = run_shot_model_cv(n_splits=int(n_splits), random_state=42)
    metrics: List[Dict[str, Any]] = []
    for model_name, row in summary_df.iterrows():
        metrics.append(
            {
                "model": model_name,
                "RMSE_mean": float(row["RMSE_mean"]),
                "RMSE_std": float(row["RMSE_std"]),
                "MAE_mean": float(row["MAE_mean"]),
                "MAE_std": float(row["MAE_std"]),
                "R2_mean": float(row["R2_mean"]),
                "R2_std": float(row["R2_std"]),
            }
        )
    return jsonable_encoder({"n_splits": int(n_splits), "metrics": metrics})


@app.get("/analysis/ml")
def ml_statistical_analysis(
    n_splits: int = Query(5, ge=2, le=10),
    min_poss: int = Query(25, ge=0, le=200),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    payload = compute_ml_analysis(
        rec.team_df,
        rec.league_df,
        n_splits=int(n_splits),
        min_poss=int(min_poss),
        force_refresh=bool(refresh),
    )
    return jsonable_encoder(payload)


@app.get("/analysis/shot-ml")
def shot_statistical_analysis(
    n_splits: int = Query(5, ge=2, le=10),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    payload = compute_shot_ml_analysis(n_splits=int(n_splits), force_refresh=bool(refresh))
    return jsonable_encoder(payload)


# ---------------------------------------------------------------------
# SportyPy visualization endpoint
# ---------------------------------------------------------------------


@app.get("/viz/playtype-zones")
def viz_playtype_zones(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    play_type: str = Query(...),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    _require_season(season)
    _require_team(our, "our")
    _require_team(opp, "opponent")
    if our == opp:
        raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

    try:
        from viz_sportypy import render_playtype_zone_png, png_bytes_to_base64
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "SportyPy visualization import failed. "
                "Make sure you installed backend deps inside the backend venv:\n"
                "  python3 -m pip install sportypy matplotlib pillow\n"
                f"Import error: {e}"
            ),
        )

    w_def = 1.0 - float(w_off)

    df = rank_playtypes_baseline(
        team_df=rec.team_df,
        league_df=rec.league_df,
        season=season,
        our_team=our,
        opp_team=opp,
        k=10,
        w_off=float(w_off),
        w_def=float(w_def),
    )

    row = df[df["PLAY_TYPE"] == play_type]
    if row.empty:
        raise HTTPException(status_code=404, detail="Play type not found in Top-K output.")

    r = row.iloc[0]
    caption = (
        f"{play_type}: Pred {float(r['PPP_PRED']):.3f} PPP. "
        f"Our(off) {float(r['PPP_OFF_SHRUNK']):.3f} vs Opp(def) {float(r['PPP_DEF_SHRUNK']):.3f}."
    )

    title = f"{our} vs {opp} • {season} • {play_type}"
    png = render_playtype_zone_png(play_type, title)

    return {"caption": caption, "image_base64": png_bytes_to_base64(png)}


# ---------------------------------------------------------------------
# Shot Intelligence endpoints (Dataset 2)
# ---------------------------------------------------------------------


@app.get("/shotplan/rank")
def rank_shotplan(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation."),
    opp: str = Query(..., description="Opponent team abbreviation."),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    w_def = float(1.0 - w_off)
    rec_shot = _get_shot_rec()
    try:
        result = rec_shot.rank(season=season, our_team=our, opp_team=opp, k=int(k), w_off=float(w_off))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return jsonable_encoder(
        {
            "season": season,
            "our_team": our,
            "opp_team": opp,
            "k": int(k),
            "w_off": float(w_off),
            "w_def": float(w_def),
            "best_shooter": None,
            "top_shot_types": result.get("top_shot_types", []),
            "top_zones": result.get("top_zones", []),
            "metadata": {"data_source": "nba_pbp_2021_present.parquet"},
        }
    )


@app.get("/viz/shot-heatmap")
def viz_shot_heatmap(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
) -> Dict[str, Any]:
    try:
        from viz_shot_heatmap import render_shot_heatmap_png, png_bytes_to_base64
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Shot heatmap import failed. Install backend deps:\n"
                "  python3 -m pip install sportypy matplotlib pillow\n"
                f"Import error: {e}"
            ),
        )

    shots_df = _get_shots_clean_df()
    title = f"{our} vs {opp} • {season}"
    png = render_shot_heatmap_png(
        shots_df=shots_df,
        season=season,
        our_team=our,
        opp_team=opp,
        shot_type=shot_type,
        zone=zone,
        title=title,
    )
    caption = f"Shot Heatmap • {our} vs {opp} • {season}"
    return {"caption": caption, "image_base64": png_bytes_to_base64(png)}


@app.get("/export/shotplan.pdf")
def export_shotplan_pdf(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
) -> StreamingResponse:
    try:
        from export_shotplan_pdf import build_shotplan_pdf
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"export_shotplan_pdf.py missing or failed to import. Error: {e}",
        )

    pdf_bytes, filename = build_shotplan_pdf(
        season=season,
        our=our,
        opp=opp,
        k=int(k),
        w_off=float(w_off),
        shot_type=shot_type,
        zone=zone,
    )
    return StreamingResponse(
        pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------
# Routers (must be registered AFTER app/rec/meta are defined)
# ---------------------------------------------------------------------

app.include_router(create_pdf_router(rec, VALID_SEASONS, VALID_TEAMS, TEAM_NAMES))