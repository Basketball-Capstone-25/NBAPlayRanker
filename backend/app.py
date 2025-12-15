"""FastAPI backend for the Basketball Game Strategy Analysis PSPI.

Key idea: one backend API that a non-basketball committee can understand.
- Data Explorer: show the cleaned dataset the models use.
- Baseline Recommender: transparent, explainable matchup ranking.
- Context + ML: shows where "AI" is actually used + how game context changes priorities.
- Model Metrics: cross-validated baseline vs ML evaluation.
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ml_context_recommender import rank_ml_with_context
from ml_models import paired_t_test_rmse, run_cv_evaluation
from state import baseline_rec, get_baseline_formula, get_meta_options, get_pipeline_info

app = FastAPI(
    title="Basketball Game Strategy API",
    description=(
        "Backend for the Basketball Game Strategy Analysis capstone."
        "\n- /data/team-playtypes: dataset preview for Data Explorer"
        "\n- /rank-plays/baseline: baseline matchup recommendations"
        "\n- /rank-plays/context-ml: ML + context recommendations"
        "\n- /metrics/baseline-vs-ml: evaluation metrics"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",  # permissive for demos
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OPTIONS = get_meta_options()
_VALID_SEASONS = set(_OPTIONS["seasons"])
_VALID_TEAMS = set(_OPTIONS["teams"])
_VALID_SIDES = {"offense", "defense"}


def _validate_matchup(season: str, our: str, opp: str) -> None:
    if season not in _VALID_SEASONS:
        raise HTTPException(400, f"Unknown season '{season}'.")
    if our not in _VALID_TEAMS:
        raise HTTPException(400, f"Unknown team '{our}'.")
    if opp not in _VALID_TEAMS:
        raise HTTPException(400, f"Unknown team '{opp}'.")
    if our == opp:
        raise HTTPException(400, "Our team and opponent must be different.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/meta/options")
def meta_options() -> Dict[str, Any]:
    return get_meta_options()


@app.get("/meta/pipeline")
def meta_pipeline() -> Dict[str, Any]:
    return get_pipeline_info()


@app.get("/meta/baseline")
def meta_baseline() -> Dict[str, Any]:
    return get_baseline_formula()


@app.get("/data/team-playtypes")
def data_team_playtypes(
    season: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    side: Optional[str] = Query(None, description="offense|defense"),
    play_type: Optional[str] = Query(None),
    min_poss: float = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, Any]:
    """Preview of the cleaned team-level dataset (NOT recommendations)."""

    df = baseline_rec.team_df.copy()

    if season is not None:
        if season not in _VALID_SEASONS:
            raise HTTPException(400, f"Unknown season '{season}'.")
        df = df[df["SEASON"] == season]

    if team is not None:
        if team not in _VALID_TEAMS:
            raise HTTPException(400, f"Unknown team '{team}'.")
        df = df[df["TEAM_ABBREVIATION"] == team]

    if side is not None:
        s = str(side).strip().lower()
        if s not in _VALID_SIDES:
            raise HTTPException(400, "side must be 'offense' or 'defense'.")
        df = df[df["SIDE"] == s]

    if play_type is not None:
        df = df[df["PLAY_TYPE"] == play_type]

    if min_poss > 0:
        df = df[df["POSS"] >= float(min_poss)]

    total_rows = int(df.shape[0])

    keep_cols = [
        "SEASON",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "PLAY_TYPE",
        "SIDE",
        "GP",
        "POSS",
        "POSS_PCT",
        "PPP",
        "FG_PCT",
        "EFG_PCT",
        "SCORE_POSS_PCT",
        "TOV_POSS_PCT",
        "SF_POSS_PCT",
        "FT_POSS_PCT",
        "PLUSONE_POSS_PCT",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].sort_values(
        ["SEASON", "TEAM_ABBREVIATION", "SIDE", "POSS"],
        ascending=[True, True, True, False],
    )

    out = df.head(limit)
    return {
        "total_rows": total_rows,
        "returned_rows": int(out.shape[0]),
        "rows": out.to_dict(orient="records"),
    }


@app.get("/data/team-playtypes.csv")
def data_team_playtypes_csv(
    season: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    side: Optional[str] = Query(None),
    play_type: Optional[str] = Query(None),
    min_poss: float = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
):
    resp = data_team_playtypes(
        season=season,
        team=team,
        side=side,
        play_type=play_type,
        min_poss=min_poss,
        limit=limit,
    )

    rows: List[Dict[str, Any]] = resp.get("rows", [])

    buf = io.StringIO()
    w = csv.writer(buf)

    if not rows:
        w.writerow(["NO_ROWS"])
    else:
        header = list(rows[0].keys())
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h) for h in header])

    buf.seek(0)

    filename_bits = ["team_playtypes"]
    if season:
        filename_bits.append(season)
    if team:
        filename_bits.append(team)
    if side:
        filename_bits.append(side)
    filename = "_".join(filename_bits) + ".csv"

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/rank-plays/baseline")
def rank_plays_baseline(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
    w_def: float = Query(0.3, ge=0, le=1),
) -> Dict[str, Any]:
    """Explainable baseline matchup ranking.

    PPP_pred = w_off * PPP_off_shrunk + w_def * PPP_def_shrunk
    """

    _validate_matchup(season, our, opp)
    if (w_off + w_def) <= 0:
        raise HTTPException(400, "w_off and w_def cannot both be 0.")

    try:
        df = baseline_rec.rank(
            season=season,
            our_team=our,
            opp_team=opp,
            k=k,
            w_off=w_off,
            w_def=w_def,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    norm = float(w_off + w_def)
    return {
        "season": season,
        "our_team": our,
        "opp_team": opp,
        "k": int(k),
        "w_off": float(w_off / norm),
        "w_def": float(w_def / norm),
        "rankings": df.to_dict(orient="records"),
    }


@app.get("/rank-plays/baseline.csv")
def rank_plays_baseline_csv(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
    w_def: float = Query(0.3, ge=0, le=1),
):
    payload = rank_plays_baseline(season=season, our=our, opp=opp, k=k, w_off=w_off, w_def=w_def)
    rows: List[Dict[str, Any]] = payload["rankings"]

    buf = io.StringIO()
    w = csv.writer(buf)

    if not rows:
        w.writerow(["NO_ROWS"])
    else:
        header = list(rows[0].keys())
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h) for h in header])

    buf.seek(0)
    filename = f"baseline_{season}_{our}_vs_{opp}_top{k}.csv"

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/rank-plays/context-ml")
def rank_plays_context_ml(
    season: str = Query(...),
    our: str = Query(...),
    opp: str = Query(...),
    margin: float = Query(...),
    period: int = Query(..., ge=1, le=5),
    time_remaining: float = Query(..., ge=0, le=720),
    k: int = Query(5, ge=1, le=10),
) -> Dict[str, Any]:
    _validate_matchup(season, our, opp)

    try:
        df = rank_ml_with_context(
            season=season,
            our_team=our,
            opp_team=opp,
            margin=float(margin),
            period=int(period),
            time_remaining_period_sec=float(time_remaining),
            k=int(k),
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))

    return {
        "season": season,
        "our_team": our,
        "opp_team": opp,
        "k": int(k),
        "margin": float(margin),
        "period": int(period),
        "time_remaining_period_sec": float(time_remaining),
        "rankings": df.to_dict(orient="records"),
    }


@lru_cache(maxsize=8)
def _cached_cv(n_splits: int) -> Dict[str, Any]:
    summary_df, fold_metrics = run_cv_evaluation(n_splits=n_splits)

    metrics: List[Dict[str, Any]] = []
    for model_name, row in summary_df.iterrows():
        metrics.append(
            {
                "model": str(model_name),
                "RMSE_mean": float(row["RMSE_mean"]),
                "RMSE_std": float(row["RMSE_std"]),
                "MAE_mean": float(row["MAE_mean"]),
                "MAE_std": float(row["MAE_std"]),
                "R2_mean": float(row["R2_mean"]),
                "R2_std": float(row["R2_std"]),
            }
        )

    t_stat, p_val = paired_t_test_rmse(fold_metrics)

    # NaN-safe handling
    t_out = None if t_stat != t_stat else float(t_stat)
    p_out = None if p_val != p_val else float(p_val)

    return {
        "n_splits": int(n_splits),
        "metrics": metrics,
        "rf_vs_baseline_t": t_out,
        "rf_vs_baseline_p": p_out,
    }


@app.get("/metrics/baseline-vs-ml")
def baseline_vs_ml_metrics(n_splits: int = Query(5, ge=2, le=10)) -> Dict[str, Any]:
    return _cached_cv(int(n_splits))
