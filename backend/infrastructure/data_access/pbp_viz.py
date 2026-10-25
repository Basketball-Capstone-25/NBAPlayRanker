from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import hashlib
import pandas as pd
from fastapi import HTTPException

from .pbp_constants import CANONICAL_PARQUET
from .pbp_io import read_parquet_cached


@dataclass
class HeatmapRequest:
    season: str
    team: str
    opp: str
    shot_type: Optional[str] = None
    zone: Optional[str] = None
    max_shots: int = 35000


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if not v or v.lower() == "all":
        return None
    return v


def _norm_code(value: Optional[str]) -> str:
    return str(value or "").strip().upper()


def _norm_text(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _stable_sample_seed(req: HeatmapRequest) -> int:
    key = "|".join(
        [
            req.season,
            req.team,
            req.opp,
            req.shot_type or "",
            req.zone or "",
            str(req.max_shots),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_canonical(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Load canonical parquet via cached loader.
    """
    try:
        return read_parquet_cached(CANONICAL_PARQUET, columns=columns)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Canonical shots parquet not found. Run the Dataset2 pipeline first so "
                "shot heatmap data exists."
            ),
        ) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Canonical parquet schema mismatch: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load canonical parquet: {e}") from e


def _validate_columns(df: pd.DataFrame) -> None:
    needed = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Canonical shots parquet is missing required columns for heatmap rendering: "
                + ", ".join(missing)
            ),
        )


def _build_available_options(df: pd.DataFrame, req: HeatmapRequest) -> Dict[str, Any]:
    same_season = df[df["season_norm"] == _norm_text(req.season)].copy()
    same_team = same_season[same_season["team_norm"] == _norm_code(req.team)].copy()
    same_matchup = same_team[same_team["opp_norm"] == _norm_code(req.opp)].copy()

    return {
        "seasons": sorted(df["season"].dropna().astype(str).unique().tolist())[:50],
        "teams": sorted(same_season["team"].dropna().astype(str).unique().tolist())[:50],
        "opponents_for_team": sorted(same_team["opp"].dropna().astype(str).unique().tolist())[:50],
        "shot_types_for_matchup": sorted(same_matchup["shot_type"].dropna().astype(str).unique().tolist())[:100],
        "zones_for_matchup": sorted(same_matchup["zone"].dropna().astype(str).unique().tolist())[:100],
    }


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["season"] = out["season"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()
    out["shot_type"] = out["shot_type"].astype(str).str.strip()
    out["zone"] = out["zone"].astype(str).str.strip()

    out["season_norm"] = out["season"].str.lower()
    out["team_norm"] = out["team"]
    out["opp_norm"] = out["opp"]
    out["shot_type_norm"] = out["shot_type"].str.lower()
    out["zone_norm"] = out["zone"].str.lower()

    return out


def _filter_shots(df: pd.DataFrame, req: HeatmapRequest) -> pd.DataFrame:
    season_norm = _norm_text(req.season)
    team_norm = _norm_code(req.team)
    opp_norm = _norm_code(req.opp)
    shot_type_norm = _norm_text(req.shot_type)
    zone_norm = _norm_text(req.zone)

    q = (df["season_norm"] == season_norm) & (df["team_norm"] == team_norm) & (df["opp_norm"] == opp_norm)

    if shot_type_norm:
        q &= df["shot_type_norm"] == shot_type_norm

    if zone_norm:
        q &= df["zone_norm"] == zone_norm

    return df.loc[q].copy()


def _make_caption(req: HeatmapRequest, counts: dict) -> str:
    parts = [f"{req.team} vs {req.opp} · {req.season}",
             f"Type: {req.shot_type or 'All'} · Zone: {req.zone or 'All'}",
             f"Matched: {counts['n_shots_total']:,} · Shown: {counts['n_shots_rendered']:,}"]
    extra = []
    if counts["n_shots_excluded_location"]:
        extra.append(f"{counts['n_shots_excluded_location']:,} uncertain locations excluded")
    if counts["n_shots_outside_halfcourt"]:
        extra.append(f"{counts['n_shots_outside_halfcourt']:,} outside half court")
    if counts["n_shots_omitted_by_cap"]:
        extra.append(f"{counts['n_shots_omitted_by_cap']:,} omitted by sample cap")
    if extra:
        parts.append(" · ".join(extra))
    return "\n".join(parts)


def _heatmap_data(*, season, team, opp, shot_type=None, zone=None, max_shots=35000):
    from infrastructure.visualization_and_export.viz_shot_heatmap import heatmap_plot_data
    req = HeatmapRequest(str(season).strip(), _norm_code(team), _norm_code(opp),
                         _clean_optional(shot_type), _clean_optional(zone), int(max_shots))
    raw = _load_canonical()
    _validate_columns(raw)
    cols = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    cols += [c for c in ("home", "location_valid") if c in raw]
    subset = raw.loc[(raw["season"] == req.season) & (raw["team"] == req.team), cols]
    df = _prepare_df(subset)
    matched = _filter_shots(df, req)
    if matched.empty:
        options = _build_available_options(df, req)
        options["seasons"] = sorted(raw["season"].dropna().unique().tolist())
        options["teams"] = sorted(raw.loc[raw["season"] == req.season, "team"].dropna().unique().tolist())
        raise HTTPException(status_code=404, detail={
            "message": "No shots matched these heatmap filters.",
            "requested": {"season": req.season, "team": req.team, "opp": req.opp,
                          "shot_type": req.shot_type, "zone": req.zone}, "available": options})
    shots = matched.rename(columns={"season":"SEASON_STR", "team":"TEAM_ABBR", "opp":"OPP_ABBR",
        "shot_type":"SHOT_TYPE", "zone":"ZONE", "x":"X", "y":"Y", "home":"HOME_FLAG",
        "location_valid":"LOCATION_VALID"})
    eligible, _, _, counts = heatmap_plot_data(shots)
    rendered = eligible if len(eligible) <= req.max_shots else eligible.sample(n=req.max_shots, random_state=_stable_sample_seed(req))
    counts.update(dataset_version=raw.attrs.get("dataset_version"), n_shots_total=len(matched), n_shots_rendered=len(rendered),
                  n_shots_omitted_by_cap=len(eligible)-len(rendered))
    return req, rendered, counts


def render_pbp_heatmap_png(*, season, team, opp, shot_type=None, zone=None, max_shots=35000) -> bytes:
    from infrastructure.visualization_and_export.viz_shot_heatmap import render_shot_heatmap_png
    req, shots, counts = _heatmap_data(season=season, team=team, opp=opp,
        shot_type=shot_type, zone=zone, max_shots=max_shots)
    return render_shot_heatmap_png(shots_df=shots, season=req.season, our_team=req.team,
        opp_team=req.opp, shot_type=req.shot_type, zone=req.zone, title=_make_caption(req, counts), matched_count=counts["n_shots_total"])


def render_pbp_heatmap_json(*, season, team, opp, shot_type=None, zone=None, max_shots=35000) -> Dict[str, Any]:
    from infrastructure.visualization_and_export.viz_shot_heatmap import render_shot_heatmap_png, png_bytes_to_base64
    req, shots, counts = _heatmap_data(season=season, team=team, opp=opp,
        shot_type=shot_type, zone=zone, max_shots=max_shots)
    caption = _make_caption(req, counts)
    png = render_shot_heatmap_png(shots_df=shots, season=req.season, our_team=req.team,
        opp_team=req.opp, shot_type=req.shot_type, zone=req.zone, title=caption, matched_count=counts["n_shots_total"])
    return {"season": req.season, "team": req.team, "opp": req.opp,
        "shot_type": req.shot_type, "zone": req.zone, "max_shots": req.max_shots,
        **counts, "caption": caption, "image_base64": png_bytes_to_base64(png)}


def render_pbp_heatmap_base64(*, season, our, opp, shot_type=None, zone=None, max_shots=35000):
    payload = render_pbp_heatmap_json(season=season, team=our, opp=opp,
        shot_type=shot_type, zone=zone, max_shots=max_shots)
    return {**payload, "n_shots": payload["n_shots_rendered"]}
