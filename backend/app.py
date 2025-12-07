from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from baseline_recommender import BaselineRecommender
from ml_models import run_cv_evaluation
from ml_context_recommender import rank_ml_with_context


app = FastAPI(
    title="Basketball Game Strategy API",
    description=(
        "Backend for the Basketball Game Strategy Analysis capstone.\n"
        "- /rank-plays/baseline: matchup play-type ranking with the baseline model.\n"
        "- /rank-plays/context-ml: matchup ranking with ML and game context.\n"
        "- /metrics/baseline-vs-ml: offline comparison of baseline vs ML models."
    ),
)

# Frontend origins allowed to call this API in the browser
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

# Seasons and team codes present in the Synergy dataset
VALID_SEASONS = {
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
}

VALID_TEAMS = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
}

# Single baseline recommender instance (reused across requests)
rec = BaselineRecommender("data/synergy_playtypes_2019_2025_players.csv")


class BaselineResponseItem(BaseModel):
    """One ranked play type from the baseline model."""

    PLAY_TYPE: str

    # Predicted efficiency for this matchup
    PPP_PRED: float
    PPP_OFF_SHRUNK: float
    PPP_DEF_SHRUNK: float
    PPP_GAP: float

    # Usage / volume
    POSS_OFF: float
    POSS_DEF: float
    POSS_PCT_OFF: float
    POSS_PCT_DEF: float

    # Shooting / scoring profile
    FG_PCT_OFF: float
    EFG_PCT_OFF: float
    EFG_PCT_DEF: float
    SCORE_POSS_PCT_OFF: float
    SCORE_POSS_PCT_DEF: float
    TOV_POSS_PCT_OFF: float
    TOV_POSS_PCT_DEF: float

    # Text explanation built in baseline_recommender.py
    RATIONALE: str


class BaselineResponse(BaseModel):
    """Response shape for /rank-plays/baseline."""

    season: str
    our_team: str
    opp_team: str
    k: int
    rankings: List[BaselineResponseItem]


class ModelMetric(BaseModel):
    """Cross-validated metrics for a single model."""

    model: str
    RMSE_mean: float
    RMSE_std: float
    MAE_mean: float
    MAE_std: float
    R2_mean: float
    R2_std: float


class ModelComparisonResponse(BaseModel):
    """Response shape for /metrics/baseline-vs-ml."""

    n_splits: int
    metrics: List[ModelMetric]


class ContextPlayItem(BaseModel):
    """One ranked play type from the context-aware ML model."""

    PLAY_TYPE: str

    PPP_PRED_ML: float
    PPP_ML: float
    PPP_DEF_SHRUNK: float
    PPP_GAP_ML: float

    CONTEXT_SCORE: float
    CONTEXT_RATIONALE: str


class ContextMLResponse(BaseModel):
    """Response shape for /rank-plays/context-ml."""

    season: str
    our_team: str
    opp_team: str
    k: int
    margin: float
    period: int
    time_remaining_period_sec: float
    rankings: List[ContextPlayItem]


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/rank-plays/baseline", response_model=BaselineResponse)
def rank_plays_baseline(
    season: str = Query(..., description="Season label, e.g. '2019-20'."),
    our: str = Query(..., description="Our team abbreviation, e.g. 'LAL'."),
    opp: str = Query(..., description="Opponent team abbreviation, e.g. 'BOS'."),
    k: int = Query(5, ge=1, le=10, description="Number of top play types to return."),
) -> BaselineResponse:
    """Baseline ranking for a given season and matchup."""
    # validate inputs
    if season not in VALID_SEASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown season '{season}'. Allowed: {sorted(VALID_SEASONS)}",
        )
    if our not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown our-team code '{our}'.",
        )
    if opp not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown opponent-team code '{opp}'.",
        )
    if our == opp:
        raise HTTPException(
            status_code=400,
            detail="Our team and opponent must be different.",
        )

    try:
        df = rec.rank(season=season, our_team=our, opp_team=opp, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    records: List[Dict[str, Any]] = df.to_dict(orient="records")

    return BaselineResponse(
        season=season,
        our_team=our,
        opp_team=opp,
        k=k,
        rankings=records,  # type: ignore[arg-type]
    )


@app.get("/rank-plays/baseline.csv")
def rank_plays_baseline_csv(
    season: str = Query(..., description="Season label, e.g. '2019-20'."),
    our: str = Query(..., description="Our team abbreviation, e.g. 'LAL'."),
    opp: str = Query(..., description="Opponent team abbreviation, e.g. 'BOS'."),
    k: int = Query(5, ge=1, le=10, description="Number of top play types to return."),
):
    """CSV version of the baseline rankings."""
    if season not in VALID_SEASONS:
        raise HTTPException(status_code=400, detail="Invalid season")
    if our not in VALID_TEAMS or opp not in VALID_TEAMS or our == opp:
        raise HTTPException(status_code=400, detail="Invalid team codes")

    try:
        df = rec.rank(season=season, our_team=our, opp_team=opp, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(df.columns)
    for _, row in df.iterrows():
        writer.writerow(list(row.values))

    buffer.seek(0)
    filename = f"baseline_{season}_{our}_vs_{opp}_top{k}.csv"

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.get("/rank-plays/context-ml", response_model=ContextMLResponse)
def rank_plays_context_ml(
    season: str = Query(..., description="Season label, e.g. '2019-20'."),
    our: str = Query(..., description="Our team abbreviation, e.g. 'LAL'."),
    opp: str = Query(..., description="Opponent team abbreviation, e.g. 'BOS'."),
    margin: float = Query(
        ...,
        description="Score margin (our score − theirs). Positive if we are leading.",
    ),
    period: int = Query(
        ...,
        ge=1,
        le=5,
        description="Game period: 1–4 for regulation, 5 for OT.",
    ),
    time_remaining: float = Query(
        ...,
        ge=0,
        le=720,
        description="Seconds remaining in the current period (0–720).",
    ),
    k: int = Query(5, ge=1, le=10, description="Number of top play types to return."),
) -> ContextMLResponse:
    """Context-aware ML ranking for a given matchup and game state."""
    if season not in VALID_SEASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown season '{season}'. Allowed: {sorted(VALID_SEASONS)}",
        )
    if our not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown our-team code '{our}'.",
        )
    if opp not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown opponent-team code '{opp}'.",
        )
    if our == opp:
        raise HTTPException(
            status_code=400,
            detail="Our team and opponent must be different.",
        )

    # OT is treated as period 5 in the urgency calculation
    clamped_period = max(1, min(period, 5))

    try:
        df = rank_ml_with_context(
            season=season,
            our_team=our,
            opp_team=opp,
            margin=margin,
            period=clamped_period,
            time_remaining_period_sec=time_remaining,
            k=k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    records: List[Dict[str, Any]] = df.to_dict(orient="records")

    return ContextMLResponse(
        season=season,
        our_team=our,
        opp_team=opp,
        k=k,
        margin=margin,
        period=period,
        time_remaining_period_sec=time_remaining,
        rankings=records,  # type: ignore[arg-type]
    )


@app.get("/metrics/baseline-vs-ml", response_model=ModelComparisonResponse)
def baseline_vs_ml_metrics(
    n_splits: int = Query(
        5,
        ge=2,
        le=10,
        description="Number of cross-validation folds.",
    ),
) -> ModelComparisonResponse:
    """Offline comparison of baseline vs ML models."""
    summary_df, _ = run_cv_evaluation(n_splits=n_splits)

    metric_items: List[ModelMetric] = []
    for model_name, row in summary_df.iterrows():
        metric_items.append(
            ModelMetric(
                model=model_name,
                RMSE_mean=float(row["RMSE_mean"]),
                RMSE_std=float(row["RMSE_std"]),
                MAE_mean=float(row["MAE_mean"]),
                MAE_std=float(row["MAE_std"]),
                R2_mean=float(row["R2_mean"]),
                R2_std=float(row["R2_std"]),
            )
        )

    return ModelComparisonResponse(n_splits=n_splits, metrics=metric_items)
