from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from baseline_recommender import BaselineRecommender
from ml_models import run_cv_evaluation


# ---------- FastAPI app setup ----------

# Create the FastAPI application object.
# This is the main entry point for all HTTP requests.
app = FastAPI(
    title="Basketball Game Strategy API",
    description=(
        "API backing the Basketball Game Strategy Analysis capstone.\n"
        "- /rank-plays/baseline: matchup-specific play-type ranking using a statistical baseline model.\n"
        "- /metrics/baseline-vs-ml: offline comparison of baseline vs ML models on historical data."
    ),
)

# These are the frontend origins that are allowed to call this API in the browser.
# (localhost:3000, 5173, etc. are typical dev ports for Next.js/Vite.)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# CORS middleware allows the browser-based frontend to talk to this backend
# without being blocked by cross-origin restrictions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # allow everything in dev (relaxed for this project)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Domain constants ----------

# Valid seasons covered by the Synergy dataset.
VALID_SEASONS = {
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
}

# Valid team abbreviations used in the CSV.
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


# ---------- Baseline recommender wiring ----------

# Instantiate the baseline recommender once, at startup, so we reuse the
# preprocessed DataFrames across requests for better performance.
rec = BaselineRecommender("data/synergy_playtypes_2019_2025_players.csv")


class BaselineResponseItem(BaseModel):
    """Single ranked play-type entry from the baseline model.

    This Pydantic model describes one row returned by the baseline ranking:
    - which play type it is
    - predicted efficiency for the matchup
    - our offense stats and the opponent's defense stats
    - usage/volume and shooting profile
    - a human-readable rationale string
    """

    PLAY_TYPE: str

    # Predicted efficiency signals (combined offense + defense)
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

    # Human-readable explanation from baseline_recommender.py
    RATIONALE: str


class BaselineResponse(BaseModel):
    """Response wrapper for the baseline play-type ranking endpoint.

    This is what the frontend gets when it calls /rank-plays/baseline:
    we echo the input parameters, and return a list of ranked play types.
    """

    season: str
    our_team: str
    opp_team: str
    k: int
    rankings: List[BaselineResponseItem]


# ---------- ML comparison wiring ----------


class ModelMetric(BaseModel):
    """Aggregated cross-validated metrics for a single model.

    For each model (Baseline, Ridge, RandomForest) we report:
    - mean and std dev of RMSE, MAE, and R² across CV folds.
    """

    model: str
    RMSE_mean: float
    RMSE_std: float
    MAE_mean: float
    MAE_std: float
    R2_mean: float
    R2_std: float


class ModelComparisonResponse(BaseModel):
    """Response for the baseline vs ML comparison endpoint.

    The frontend uses this to build the metrics table on the Model Performance page.
    """

    n_splits: int
    metrics: List[ModelMetric]


# ---------- Routes ----------


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """Simple healthcheck so you can confirm the API is running.

    The frontend (or you in a browser) can call /health to see {"status": "ok"}.
    """
    return {"status": "ok"}


@app.get("/rank-plays/baseline", response_model=BaselineResponse)
def rank_plays_baseline(
    season: str = Query(..., description="Season label, e.g. '2019-20'."),
    our: str = Query(..., description="Our team abbreviation, e.g. 'LAL'."),
    opp: str = Query(..., description="Opponent team abbreviation, e.g. 'BOS'."),
    k: int = Query(5, ge=1, le=10, description="How many top play-types to return."),
) -> BaselineResponse:
    """Rank offensive play types for a specific matchup using the baseline model.

    This uses the rule-based baseline defined in baseline_recommender.py:
    - Aggregates Synergy play-type stats to team offense and defense tables.
    - Blends team PPP with league-average PPP using reliability weights.
    - Combines our offense and the opponent's defense into a predicted PPP score.
    - Returns the top-k play types with a human-readable rationale string.
    """
    # --- Input validation to keep the API robust ---
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

    # Call the baseline recommender; convert any ValueError into an HTTP error.
    try:
        df = rec.rank(season=season, our_team=our, opp_team=opp, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert the pandas DataFrame into a list of dicts for Pydantic to parse.
    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    # Pydantic will coerce each dict into BaselineResponseItem automatically.
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
    k: int = Query(5, ge=1, le=10, description="How many top play-types to return."),
):
    """CSV download version of the baseline rankings for coaches/analysts.

    This endpoint is used by the UI "Download Top-K as CSV" button and can also
    be opened directly in Excel. It reuses the same validation and ranking logic
    as /rank-plays/baseline.
    """
    # Basic validation (same as JSON endpoint)
    if season not in VALID_SEASONS:
        raise HTTPException(status_code=400, detail="Invalid season")
    if our not in VALID_TEAMS or opp not in VALID_TEAMS or our == opp:
        raise HTTPException(status_code=400, detail="Invalid team codes")

    try:
        df = rec.rank(season=season, our_team=our, opp_team=opp, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert the DataFrame to CSV in-memory (no temp file on disk).
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    # Header row
    writer.writerow(df.columns)
    # Data rows
    for _, row in df.iterrows():
        writer.writerow(list(row.values))

    buffer.seek(0)
    filename = f"baseline_{season}_{our}_vs_{opp}_top{k}.csv"
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/metrics/baseline-vs-ml", response_model=ModelComparisonResponse)
def baseline_vs_ml_metrics(
    n_splits: int = Query(
        5,
        ge=2,
        le=10,
        description="Number of cross-validation folds for the offline comparison.",
    ),
) -> ModelComparisonResponse:
    """Compare the simple statistical baseline to the ML models on historical data.

    This endpoint runs an offline experiment that:
    - Builds a team-season-playtype offense dataset from the same Synergy CSV.
    - Uses league-average PPP as the simple statistical baseline.
    - Trains Ridge regression and RandomForest models on the same features.
    - Evaluates all models with K-fold cross-validation using RMSE, MAE, and R².

    The response contains one row per model with mean and standard deviation
    for each metric across folds. It is intended for analysis / defence, not
    for live in-game ranking.
    """
    # Run the cross-validated evaluation defined in ml_models.py
    summary_df, _ = run_cv_evaluation(n_splits=n_splits)

    # Convert the summary DataFrame into a list of ModelMetric objects.
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
