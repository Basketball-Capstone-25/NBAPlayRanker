"""Shot Plan helpers for the /pbp router."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from domain.baseline_recommendation import ShotBaselineRecommender

_REC: Optional[ShotBaselineRecommender] = None


def _prime_meta(rec: ShotBaselineRecommender) -> None:
    """
    Compute meta options once from the small aggregated parquet.
    This keeps /pbp/meta/options fast and avoids touching the large canonical parquet.
    """
    if getattr(rec, "available_seasons", None) and getattr(rec, "available_teams", None):
        return

    try:
        df = rec.agg_df

        seasons = sorted(df["SEASON_STR"].dropna().unique().tolist()) if "SEASON_STR" in df.columns else []
        teams = sorted(df["TEAM_ABBR"].dropna().unique().tolist()) if "TEAM_ABBR" in df.columns else []

        shot_types = []
        if "SHOT_TYPE" in df.columns:
            shot_types = sorted(df["SHOT_TYPE"].dropna().unique().tolist())

        zones = []
        if "ZONE" in df.columns:
            zones = sorted(df["ZONE"].dropna().unique().tolist())

        rec.available_seasons = seasons  # type: ignore[attr-defined]
        rec.available_teams = teams  # type: ignore[attr-defined]
        rec.shot_types = shot_types  # type: ignore[attr-defined]
        rec.zones = zones  # type: ignore[attr-defined]
    except Exception:
        # If meta fails for any reason, don't crash startup; endpoints will still work.
        rec.available_seasons = []  # type: ignore[attr-defined]
        rec.available_teams = []  # type: ignore[attr-defined]
        rec.shot_types = []  # type: ignore[attr-defined]
        rec.zones = []  # type: ignore[attr-defined]


def _get_rec() -> ShotBaselineRecommender:
    global _REC
    if _REC is None:
        try:
            _REC = ShotBaselineRecommender()
            _prime_meta(_REC)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Shot aggregates not found. Run Phase 1 build first:\n"
                    "  python backend/data/etl/build_pbp_pipeline.py --force"
                ),
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to init shot recommender: {e}") from e
    return _REC


def get_shotplan_meta_options() -> Dict[str, Any]:
    """Options for PBP dropdowns (seasons/teams/shot types/zones).

    Returned shape is frontend-friendly and stable:
      {
        "seasons": [...],
        "teams": [...],
        "shotTypes": [...],
        "zones": [...]
      }
    """
    rec = _get_rec()
    seasons = list(getattr(rec, "available_seasons", []) or [])
    teams = list(getattr(rec, "available_teams", []) or [])
    shot_types = list(getattr(rec, "shot_types", []) or [])
    zones = list(getattr(rec, "zones", []) or [])
    return {"seasons": seasons, "teams": teams, "shotTypes": shot_types, "zones": zones}


def get_shotplan_json(
    *,
    season: str,
    our: str,
    opp: str,
    k: int = 5,
    w_off: float = 0.7,
) -> Dict[str, Any]:
    """Rank shot types/zones for offense vs opponent defense.

    Wraps ShotBaselineRecommender.rank() but returns a stable JSON payload.
    """
    rec = _get_rec()

    seasons = set(getattr(rec, "available_seasons", []) or [])
    teams = set(getattr(rec, "available_teams", []) or [])

    # Fast validation (only if meta exists)
    if seasons and season not in seasons:
        raise HTTPException(status_code=400, detail=f"Unknown season '{season}'.")
    if teams:
        if our not in teams:
            raise HTTPException(status_code=400, detail=f"Unknown team '{our}'.")
        if opp not in teams:
            raise HTTPException(status_code=400, detail=f"Unknown opponent '{opp}'.")

    try:
        k_int = int(k)
        w_off_f = float(w_off)
        res = rec.rank(season=season, our_team=our, opp_team=opp, k=k_int, w_off=w_off_f)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shotplan ranking failed: {e}") from e

    # ShotBaselineRecommender.rank returns a dict in your codebase
    payload: Dict[str, Any] = dict(res) if isinstance(res, dict) else (getattr(res, "__dict__", {}) or {})
    payload.update(
        {
            "season": season,
            "our_team": our,
            "opp_team": opp,
            "k": int(k),
            "w_off": float(w_off),
            "w_def": float(1.0 - float(w_off)),
        }
    )
    return payload
