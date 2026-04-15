# backend/pbp_shotplan.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from shot_baseline_recommender import ShotBaselineRecommender, _rank_level

# Singleton (fast startup + no repeated parquet reads)
_REC: Optional[ShotBaselineRecommender] = None


def _norm_code(value: str) -> str:
    return str(value or "").strip().upper()


def _prime_meta(rec: ShotBaselineRecommender) -> None:
    """
    Compute meta options once from the small aggregated parquet.
    This keeps /pbp/meta/options fast and avoids touching the large canonical parquet.
    """
    if getattr(rec, "available_seasons", None) and getattr(rec, "available_teams", None):
        return

    try:
        df = rec.agg_df

        seasons = sorted(df["SEASON_STR"].dropna().astype(str).unique().tolist()) if "SEASON_STR" in df.columns else []
        teams = sorted(df["TEAM_ABBR"].dropna().astype(str).unique().tolist()) if "TEAM_ABBR" in df.columns else []

        shot_types: List[str] = []
        if "SHOT_TYPE" in df.columns:
            shot_types = sorted(
                df.loc[df["LEVEL"] == "shot_type", "SHOT_TYPE"].dropna().astype(str).unique().tolist()
            )

        zones: List[str] = []
        if "ZONE" in df.columns:
            zones = sorted(
                df.loc[df["LEVEL"] == "zone", "ZONE"].dropna().astype(str).unique().tolist()
            )

        rec.available_seasons = seasons  # type: ignore[attr-defined]
        rec.available_teams = teams  # type: ignore[attr-defined]
        rec.shot_types = shot_types  # type: ignore[attr-defined]
        rec.zones = zones  # type: ignore[attr-defined]
    except Exception:
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
                    "Shot aggregates not found. Build Dataset2 aggregates first so Shot Plan can run."
                ),
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize shot recommender: {e}") from e
    return _REC


def get_shotplan_meta_options() -> Dict[str, Any]:
    """
    Options for Dataset2 dropdowns.

    Returned shape:
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


def _safe_top_pairs(
    *,
    rec: ShotBaselineRecommender,
    season: str,
    our_team: str,
    opp_team: str,
    k: int,
    w_off: float,
    w_def: float,
) -> List[Dict[str, Any]]:
    """
    Optional pair-level ranking from LEVEL == shot_type_zone.
    If that level is unavailable for any reason, return [] instead of failing the whole endpoint.
    """
    try:
        pair_df = _rank_level(
            level="shot_type_zone",
            agg_df=rec.agg_df,
            league_df=rec.league_df,
            season=season,
            our_team=our_team,
            opp_team=opp_team,
            k=k,
            w_off=w_off,
            w_def=w_def,
        )
        if pair_df is None or pair_df.empty:
            return []
        return pair_df.replace({float("nan"): None}).to_dict(orient="records")
    except Exception:
        return []


def _best_shooter_stub(
    *,
    top_shot_types: List[Dict[str, Any]],
    top_zones: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Dataset2 aggregate tables are team-level, not player-level.
    Return a lightweight summary object instead of pretending we have shooter-level rankings.
    """
    if not top_shot_types and not top_zones:
        return None

    return {
        "PLAYER_NAME": "Team-level baseline only",
        "player_name": "Team-level baseline only",
        "note": "Shot Plan is currently built from team/opponent aggregate shot data, not shooter-level player splits.",
        "best_shot_type": top_shot_types[0].get("SHOT_TYPE") if top_shot_types else None,
        "best_zone": top_zones[0].get("ZONE") if top_zones else None,
    }


def get_shotplan_json(
    *,
    season: str,
    our: str,
    opp: str,
    k: int = 5,
    w_off: float = 0.7,
) -> Dict[str, Any]:
    """
    Rank shot types / zones for our offense vs opponent defense and return a stable,
    frontend-friendly payload for the Shot Plan page.
    """
    rec = _get_rec()

    season = str(season).strip()
    our = _norm_code(our)
    opp = _norm_code(opp)

    seasons = set(getattr(rec, "available_seasons", []) or [])
    teams = set(getattr(rec, "available_teams", []) or [])

    if not season:
        raise HTTPException(status_code=400, detail="Season is required.")
    if not our:
        raise HTTPException(status_code=400, detail="Our team is required.")
    if not opp:
        raise HTTPException(status_code=400, detail="Opponent team is required.")
    if our == opp:
        raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

    try:
        k_int = int(k)
        w_off_f = float(w_off)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numeric parameters: {e}") from e

    if not (1 <= k_int <= 10):
        raise HTTPException(status_code=400, detail="k must be between 1 and 10.")
    if not (0.0 <= w_off_f <= 1.0):
        raise HTTPException(status_code=400, detail="w_off must be between 0 and 1.")

    if seasons and season not in seasons:
        raise HTTPException(status_code=400, detail=f"Unknown season '{season}'.")
    if teams:
        if our not in teams:
            raise HTTPException(status_code=400, detail=f"Unknown team '{our}'.")
        if opp not in teams:
            raise HTTPException(status_code=400, detail=f"Unknown opponent '{opp}'.")

    w_def_f = float(1.0 - w_off_f)

    try:
        res = rec.rank(
            season=season,
            our_team=our,
            opp_team=opp,
            k=k_int,
            w_off=w_off_f,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shot Plan ranking failed: {e}") from e

    payload: Dict[str, Any] = dict(res) if isinstance(res, dict) else (getattr(res, "__dict__", {}) or {})

    top_shot_types = payload.get("top_shot_types") if isinstance(payload.get("top_shot_types"), list) else []
    top_zones = payload.get("top_zones") if isinstance(payload.get("top_zones"), list) else []
    top_pairs = _safe_top_pairs(
        rec=rec,
        season=season,
        our_team=our,
        opp_team=opp,
        k=k_int,
        w_off=w_off_f,
        w_def=w_def_f,
    )

    notes: List[str] = [
        "Rankings are matchup-specific and combine your offense with the opponent's defense-allowed profile.",
        "Shot type and zone tables are team-level baseline recommendations built from Dataset2 aggregates.",
    ]
    if top_pairs:
        notes.append("Pair-level shot type + zone combinations are also available for tighter tactical filtering.")
    else:
        notes.append("Pair-level shot type + zone combinations were not available from the current aggregate build.")

    return {
        "season": season,
        "our_team": our,
        "opp_team": opp,
        "k": k_int,
        "w_off": w_off_f,
        "w_def": w_def_f,
        "top_shot_types": top_shot_types,
        "top_zones": top_zones,
        "top_pairs": top_pairs,
        "best_shooter": _best_shooter_stub(
            top_shot_types=top_shot_types,
            top_zones=top_zones,
        ),
        "metadata": {
            "available_seasons_count": len(seasons),
            "available_teams_count": len(teams),
            "source": "Dataset2 shot aggregates",
        },
        "notes": notes,
    }