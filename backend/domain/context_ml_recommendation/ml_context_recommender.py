#ml_context_recommender.py

from __future__ import annotations

"""Context-aware ML recommendation pipeline."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from domain.baseline_recommendation import BaselineRecommender

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SYNERGY_CSV_PATH = DATA_DIR / "synergy_playtypes_2019_2025_players.csv"
ML_PRED_PATH = DATA_DIR / "ml_offense_ppp_predictions.csv"

_rec = BaselineRecommender(str(SYNERGY_CSV_PATH))

QUICK_WEIGHTS: Dict[str, float] = {
    "Spotup": 1.0,
    "OffScreen": 0.8,
    "Cut": 0.6,
    "Isolation": 0.4,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def total_time_remaining(period: int, time_remaining_period_sec: float) -> float:
    """Total seconds remaining in regulation (4x12). OT is treated as 0 remaining."""
    if period >= 5:
        return 0.0
    total_reg = 4 * 12 * 60  # 2880
    elapsed = (period - 1) * 12 * 60 + (12 * 60 - time_remaining_period_sec)
    return float(max(0.0, total_reg - elapsed))

def compute_context_factors(margin: float, period: int, time_remaining_period_sec: float) -> Tuple[float, float, float]:
    """Compute the 3 simple context factors used in the app:"""
    T_left = total_time_remaining(period, time_remaining_period_sec)

    late_window = 180.0  # last 3 minutes
    late_game_factor = _clamp((late_window - T_left) / late_window, 0.0, 1.0)

    trailing_factor = _clamp((-margin) / 10.0, 0.0, 1.0) if margin < 0 else 0.0
    leading_factor = _clamp((margin) / 10.0, 0.0, 1.0) if margin > 0 else 0.0

    return float(late_game_factor), float(trailing_factor), float(leading_factor)

def label_context(late_game_factor: float, trailing_factor: float, leading_factor: float) -> str:
    if late_game_factor >= 0.5 and trailing_factor > 0:
        return "Late & trailing"
    if late_game_factor >= 0.5 and leading_factor > 0:
        return "Late & leading"
    return "Normal context"

# Core builders

def build_ml_matchup_table(
    season: str,
    our_team: str,
    opp_team: str,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """Build the matchup table that contains BOTH:"""
    team_df = _rec.team_df.copy()
    league_df = _rec.league_df.copy()

    if not ML_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing ML prediction file: {ML_PRED_PATH}")
    ml_df = pd.read_csv(ML_PRED_PATH)

    # Attach PPP_ML to team offense rows using (season, team, play type)
    team_df = team_df.merge(
        ml_df,
        on=["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"],
        how="left",
    )

    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty:
        raise ValueError(f"No offensive data for {our_team} in season {season}")
    if deff.empty:
        raise ValueError(f"No defensive data for {opp_team} in season {season}")

    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP"]].rename(columns={"PPP": "PPP_LEAGUE_OFF"})

    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP"]].rename(columns={"PPP": "PPP_LEAGUE_DEF"})

    deff_subset = deff[
        [
            "PLAY_TYPE",
            "PPP",
            "POSS",
            "POSS_PCT",
            "RELIABILITY_WEIGHT",
            "EFG_PCT",
            "TOV_POSS_PCT",
        ]
    ].copy()

    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Shrink our offense PPP and opponent defense PPP toward league anchors
    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged["PPP_LEAGUE_OFF"]
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]

    merged["PPP_BASELINE"] = (
        float(w_off) * merged["PPP_OFF_SHRUNK"]
        + float(w_def) * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Keep only play types with ML predictions
    merged = merged[merged["PPP_ML"].notna()].copy()
    if merged.empty:
        raise ValueError("No ML predictions available for this matchup (PPP_ML is missing).")

    merged["PPP_ML_BLEND"] = (
        float(w_off) * merged["PPP_ML"]
        + float(w_def) * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    return merged.reset_index(drop=True)

def apply_context_adjustments(
    df: pd.DataFrame,
    margin: float,
    period: int,
    time_remaining_period_sec: float,
) -> pd.DataFrame:
    """Apply small, transparent context adjustments and compute PPP_CONTEXT."""
    out = df.copy()

    late_game_factor, trailing_factor, leading_factor = compute_context_factors(
        margin=margin,
        period=period,
        time_remaining_period_sec=time_remaining_period_sec,
    )

    # Quick priority mapping (unknown play types -> 0)
    out["QUICK_PRIORITY"] = out["PLAY_TYPE"].map(QUICK_WEIGHTS).fillna(0.0)

    # Baselines for normalization
    avg_efg = float(out["EFG_PCT_OFF"].mean()) if "EFG_PCT_OFF" in out.columns else 0.0
    avg_tov = float(out["TOV_POSS_PCT_OFF"].mean()) if "TOV_POSS_PCT_OFF" in out.columns else 0.0

    # Bonuses/penalties (small + explainable)
    out["BONUS_QUICK"] = (
        0.04 * late_game_factor * (0.3 + trailing_factor) * out["QUICK_PRIORITY"]
    )

    out["BONUS_SCORE"] = (
        0.25
        * late_game_factor
        * trailing_factor
        * np.maximum(0.0, out["EFG_PCT_OFF"] - avg_efg)
    )

    out["PENALTY_PROTECT"] = (
        0.30
        * late_game_factor
        * leading_factor
        * np.maximum(0.0, out["TOV_POSS_PCT_OFF"] - avg_tov)
    )

    out["CONTEXT_ADJ"] = out["BONUS_QUICK"] + out["BONUS_SCORE"] - out["PENALTY_PROTECT"]
    out["PPP_CONTEXT"] = out["PPP_ML_BLEND"] + out["CONTEXT_ADJ"]
    out["DELTA_VS_BASELINE"] = out["PPP_CONTEXT"] - out["PPP_BASELINE"]

    ctx_label = label_context(late_game_factor, trailing_factor, leading_factor)
    out["CONTEXT_LABEL"] = ctx_label

    # Expose the exact factors (committee can verify logic)
    out["LATE_GAME_FACTOR"] = float(late_game_factor)
    out["TRAILING_FACTOR"] = float(trailing_factor)
    out["LEADING_FACTOR"] = float(leading_factor)

    def rationale(row: pd.Series) -> str:
        return (
            f"{row['PLAY_TYPE']}: ML base {row['PPP_ML_BLEND']:.3f}, "
            f"adj {row['CONTEXT_ADJ']:+.3f} ({ctx_label}). "
            f"Late={late_game_factor:.2f}, trailing={trailing_factor:.2f}, leading={leading_factor:.2f}."
        )

    out["RATIONALE"] = out.apply(rationale, axis=1)
    return out

def rank_ml_with_context(
    season: str,
    our_team: str,
    opp_team: str,
    margin: float,
    period: int,
    time_remaining_period_sec: float,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """Full AI use case ranking:"""
    df = build_ml_matchup_table(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        w_off=w_off,
        w_def=w_def,
    )
    df = apply_context_adjustments(
        df=df,
        margin=margin,
        period=period,
        time_remaining_period_sec=time_remaining_period_sec,
    )
    df = df.sort_values(["PPP_CONTEXT", "PPP_ML_BLEND"], ascending=False).reset_index(drop=True)
    return df.head(k)
