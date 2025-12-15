"""backend/ml_context_recommender.py

Context + ML recommender (this is the "AI use case" the committee asked for).

What this module does:
1) Starts from the SAME matchup table as the baseline model (our offense vs opponent defense).
2) Uses an ML-predicted offensive PPP for each play type (PPP_ML) instead of using historical PPP only.
3) Applies a small, transparent "context adjustment" based on game state:
   - late game + trailing -> boost high-efficiency options + familiar options
   - late game + leading  -> penalize turnover-prone options (protect the lead)

Important:
- We do NOT hide the logic. The endpoint returns the breakdown columns so the UI can
  show exactly why rankings change.
- This is not meant to be a perfect NBA coaching brain; it's a defendable, reproducible,
  context-aware decision support demo (capstone scope).

Inputs:
- season, our_team, opp_team (same as baseline)
- margin (our_score - opponent_score; negative means trailing)
- period (1-4, 5 for OT)
- time_remaining_period_sec (seconds remaining in the current period)
- k (top-K results)

Output:
A DataFrame sorted by PPP_CONTEXT (ML+context) with explainability columns:
- PPP_BASELINE (baseline predicted PPP)
- PPP_ML_BLEND (ML-based predicted PPP, before context)
- CONTEXT_ADJ (total context adjustment)
- PPP_CONTEXT (final score used for ranking)
- plus component columns and a human-readable RATIONALE string
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from state import baseline_rec, ml_pred_df


# -----------------------------
# Context interpretation helpers
# -----------------------------

def _late_game_factor(period: int, time_remaining: float) -> float:
    """0..1 late-game factor.

    Simple rule:
    - Only ramps up in Q4 or OT.
    - Stronger when <= 3 minutes remaining in the period.
    """
    if period < 4:
        return 0.0
    # Clamp 0..1 where 180 sec => 1, 720 sec => 0
    return float(np.clip((180.0 - float(time_remaining)) / 180.0, 0.0, 1.0))


def _score_pressure(margin: float) -> Tuple[float, float]:
    """Return (trailing_factor, leading_factor) in {0,1} based on margin."""
    # Negative margin means we are trailing.
    if margin <= -4:
        return 1.0, 0.0
    if margin >= 4:
        return 0.0, 1.0
    return 0.0, 0.0  # close game


def _zscore(series: pd.Series) -> pd.Series:
    """Compute z-score safely (returns 0 if std=0 or missing)."""
    s = pd.to_numeric(series, errors="coerce")
    mu = float(s.mean()) if s.notna().any() else 0.0
    sd = float(s.std(ddof=0)) if s.notna().any() else 0.0
    if sd <= 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


# --------------------------------------------
# Core: build matchup table + compute rankings
# --------------------------------------------

def _build_matchup_table(season: str, our_team: str, opp_team: str) -> pd.DataFrame:
    """Build a matchup table with BOTH baseline fields and ML offensive predictions.

    This table is similar to baseline_recommender.rank() but includes:
    - PPP_ML for our offensive play types
    - PPP_DEF_SHRUNK and opponent defensive tendencies
    """

    # Pull offense and defense rows from cached baseline tables.
    off = baseline_rec.offense_tbl.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team"
    ).copy()

    deff = baseline_rec.defense_tbl.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No offense/defense rows available for this matchup.")

    # Opponent defense subset
    deff_sub = deff[
        [
            "PLAY_TYPE",
            "PPP",
            "POSS",
            "POSS_PCT",
            "RELIABILITY_WEIGHT",
            "EFG_PCT",
            "TOV_POSS_PCT",
            "SCORE_POSS_PCT",
            "FT_POSS_PCT",
        ]
    ].copy()

    # Merge on PLAY_TYPE
    m = off.merge(deff_sub, on="PLAY_TYPE", suffixes=("_OFF", "_DEF"))

    # League baselines (for shrinkage)
    league_off = baseline_rec.league_df.query("SEASON == @season and SIDE == 'offense'")[
        ["PLAY_TYPE", "PPP"]
    ].rename(columns={"PPP": "PPP_LEAGUE_OFF"})

    league_def = baseline_rec.league_df.query("SEASON == @season and SIDE == 'defense'")[
        ["PLAY_TYPE", "PPP"]
    ].rename(columns={"PPP": "PPP_LEAGUE_DEF"})

    m = m.merge(league_off, on="PLAY_TYPE", how="left")
    m = m.merge(league_def, on="PLAY_TYPE", how="left")

    # Shrinkage using team-level reliability weights.
    rel_off = pd.to_numeric(m["RELIABILITY_WEIGHT_OFF"], errors="coerce").fillna(0.0)
    rel_def = pd.to_numeric(m["RELIABILITY_WEIGHT_DEF"], errors="coerce").fillna(0.0)

    m["PPP_OFF_SHRUNK"] = rel_off * m["PPP_OFF"] + (1 - rel_off) * m["PPP_LEAGUE_OFF"]
    m["PPP_DEF_SHRUNK"] = rel_def * m["PPP_DEF"] + (1 - rel_def) * m["PPP_LEAGUE_DEF"]

    # ---- ML predictions join (our offense) ----
    if ml_pred_df is None:
        raise FileNotFoundError(
            "ML predictions file is missing. Expected backend/data/ml_offense_ppp_predictions.csv. "
            "Run backend/ml_models.py to generate it."
        )

    pred = ml_pred_df.copy()
    pred = pred.query("SEASON == @season and TEAM_ABBREVIATION == @our_team")[
        ["PLAY_TYPE", "PPP_ML"]
    ].drop_duplicates()

    m = m.merge(pred, on="PLAY_TYPE", how="left")

    # If a play type is missing an ML prediction, fall back to shrunk historical offense PPP.
    m["PPP_ML"] = pd.to_numeric(m["PPP_ML"], errors="coerce")
    m["PPP_ML_FILLED"] = m["PPP_ML"].fillna(m["PPP_OFF_SHRUNK"])

    return m


def rank_ml_with_context(
    season: str,
    our_team: str,
    opp_team: str,
    margin: float,
    period: int,
    time_remaining_period_sec: float,
    k: int = 5,
    w_off_ml: float = 0.75,
    w_def: float = 0.25,
) -> pd.DataFrame:
    """Rank play types using ML + context.

    Steps:
    1) Baseline (for comparison): uses PPP_OFF_SHRUNK + PPP_DEF_SHRUNK.
    2) ML blend: uses PPP_ML (predicted offense PPP) + PPP_DEF_SHRUNK.
    3) Context adjustment: small boosts/penalties based on late-game + score state.
    """

    if not (1 <= k <= 10):
        raise ValueError("k must be between 1 and 10.")
    if period < 1 or period > 5:
        raise ValueError("period must be 1..5 (5=OT).")
    if time_remaining_period_sec < 0 or time_remaining_period_sec > 720:
        raise ValueError("time_remaining_period_sec must be between 0 and 720.")
    if w_off_ml < 0 or w_def < 0 or (w_off_ml + w_def) <= 0:
        raise ValueError("w_off_ml and w_def must be non-negative and not both zero.")

    # Normalize weights
    s = float(w_off_ml + w_def)
    w_off_ml = float(w_off_ml / s)
    w_def = float(w_def / s)

    m = _build_matchup_table(season=season, our_team=our_team, opp_team=opp_team)

    # ---------------------------
    # 1) Baseline for comparison
    # ---------------------------
    w_off_baseline = 0.70
    w_def_baseline = 0.30
    m["PPP_BASELINE"] = w_off_baseline * m["PPP_OFF_SHRUNK"] + w_def_baseline * m["PPP_DEF_SHRUNK"]

    # ---------------------------
    # 2) ML blend score (pre-context)
    # ---------------------------
    m["PPP_ML_BLEND"] = w_off_ml * m["PPP_ML_FILLED"] + w_def * m["PPP_DEF_SHRUNK"]

    # ---------------------------
    # 3) Context adjustment (transparent + small)
    # ---------------------------
    late = _late_game_factor(period=int(period), time_remaining=float(time_remaining_period_sec))
    trailing_factor, leading_factor = _score_pressure(float(margin))

    # Features we use for simple context logic (all from offense rows)
    # - familiarity: POSS_PCT_OFF (more practiced -> easier to run under pressure)
    # - efficiency proxy: EFG_PCT_OFF (higher -> better scoring efficiency)
    # - turnover risk: TOV_POSS_PCT_OFF (higher -> risky when protecting a lead)
    poss_pct_z = _zscore(m.get("POSS_PCT_OFF", pd.Series([0.0] * len(m))))
    efg_z = _zscore(m.get("EFG_PCT_OFF", pd.Series([0.0] * len(m))))
    tov_z = _zscore(m.get("TOV_POSS_PCT_OFF", pd.Series([0.0] * len(m))))

    # Small weights so adjustments are interpretable and do NOT dominate PPP.
    # Think of this as +/- a few hundredths of PPP.
    quick_bonus = late * poss_pct_z * 0.020
    score_bonus = late * trailing_factor * efg_z * 0.030
    protect_penalty = late * leading_factor * tov_z * 0.025  # penalize high turnover tendency

    m["LATE_GAME_FACTOR"] = late
    m["TRAILING_FACTOR"] = trailing_factor
    m["LEADING_FACTOR"] = leading_factor

    m["BONUS_QUICK"] = quick_bonus
    m["BONUS_SCORE"] = score_bonus
    m["PENALTY_PROTECT"] = protect_penalty

    m["CONTEXT_ADJ"] = m["BONUS_QUICK"] + m["BONUS_SCORE"] - m["PENALTY_PROTECT"]
    m["PPP_CONTEXT"] = m["PPP_ML_BLEND"] + m["CONTEXT_ADJ"]

    # Delta compared to baseline (what changed)
    m["DELTA_VS_BASELINE"] = m["PPP_CONTEXT"] - m["PPP_BASELINE"]

    # ---------------------------
    # Human-friendly label + rationale
    # ---------------------------
    if late <= 0.0:
        context_label = "Normal context (early/mid game)"
    else:
        if trailing_factor > 0:
            context_label = "Late game, trailing (prioritize scoring)"
        elif leading_factor > 0:
            context_label = "Late game, leading (protect the lead)"
        else:
            context_label = "Late game, close score (balanced choices)"

    def rationale(row: pd.Series) -> str:
        d = float(row["DELTA_VS_BASELINE"])
        sign = "+" if d >= 0 else ""
        return (
            f"{context_label}. "
            f"Baseline={row['PPP_BASELINE']:.3f}, ML={row['PPP_ML_BLEND']:.3f}, "
            f"Adj={row['CONTEXT_ADJ']:.3f} => Final={row['PPP_CONTEXT']:.3f} "
            f"({sign}{d:.3f} vs baseline)."
        )

    m["CONTEXT_LABEL"] = context_label
    m["RATIONALE"] = m.apply(rationale, axis=1)

    # Sort by final context PPP, then by our possessions (more reliable/common)
    m = m.sort_values(["PPP_CONTEXT", "POSS_OFF"], ascending=[False, False])

    # Return top-K with clear fields (used by context/page.tsx)
    cols = [
        "PLAY_TYPE",
        "PPP_CONTEXT",
        "PPP_ML_BLEND",
        "PPP_BASELINE",
        "DELTA_VS_BASELINE",
        "PPP_ML",
        "PPP_ML_FILLED",
        "PPP_DEF_SHRUNK",
        "PPP_OFF_SHRUNK",
        "POSS_OFF",
        "POSS_PCT_OFF",
        "EFG_PCT_OFF",
        "TOV_POSS_PCT_OFF",
        "LATE_GAME_FACTOR",
        "TRAILING_FACTOR",
        "LEADING_FACTOR",
        "BONUS_QUICK",
        "BONUS_SCORE",
        "PENALTY_PROTECT",
        "CONTEXT_ADJ",
        "CONTEXT_LABEL",
        "RATIONALE",
    ]

    cols = [c for c in cols if c in m.columns]
    top = m.head(int(k))[cols].reset_index(drop=True)

    return top
