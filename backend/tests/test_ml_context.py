from __future__ import annotations

"""
backend/test_ml_context.py

Small driver script for the CURRENT context-aware ML recommender.

Why this file exists:
- sanity-check the live context-aware ranking flow from the command line
- show how guardrails/defaults behave for ambiguous prompts
- show how score/time context changes ranking between early and late game
- use the CURRENT public API from ml_context_recommender.py

This version removes references to old functions that no longer exist
(e.g. add_playtype_flags, compute_urgencies) and uses the current
recommender pipeline instead.
"""

from typing import Dict, List

import pandas as pd

from ml_context_recommender import (
    apply_context_adjustments,
    build_ml_matchup_table,
    compute_context_factors,
    rank_ml_with_context,
    recommender_health,
    sanitize_context_for_ranking,
    total_time_remaining,
    validate_context_guardrails,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def main() -> None:
    season = "2019-20"
    our_team = "LAL"
    opp_team = "BOS"

    _print_header("1) Recommender health + guardrail validation")
    print("Recommender health:")
    print(recommender_health())

    print("\nGuardrail validation:")
    print(validate_context_guardrails())

    _print_header("2) Neutral sanitized context")
    neutral_context = sanitize_context_for_ranking(
        {
            "need": "quick2",
            "needs": ["quick2"],
            "defense_style": "switch",
            "after_timeout": True,
            "special_situations": ["after_timeout"],
            "preferred_play_families": ["ball_screen", "cut"],
            "shot_clock": 7,
            "late_clock": True,
            "offense_bias": 0.75,
            "defense_bias": 0.25,
            "parser_version": "3.0.0",
            "nlp_pipeline_version": "1.0.0",
        },
        margin=-4,
        period=4,
        time_remaining=28,
    )
    print(neutral_context)

    _print_header("3) Ambiguous prompt fallback / defaults")
    fallback_context = sanitize_context_for_ranking(
        {"raw_text": "Just win the game"},
        margin=None,
        period=None,
        time_remaining=None,
    )
    print(fallback_context)

    _print_header("4) Build matchup table and inspect current play-family mapping")
    matchup_df = build_ml_matchup_table(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        context=neutral_context,
    )

    print(
        _safe_cols(
            matchup_df,
            [
                "PLAY_TYPE",
                "PLAY_FAMILY",
                "PPP_ML_BLEND",
                "PPP_BASELINE",
                "EFFECTIVE_W_OFF",
                "EFFECTIVE_W_DEF",
            ],
        ).to_string(index=False)
    )

    _print_header("5) Time remaining + context factors")
    t_early = total_time_remaining(period=1, time_remaining_period_sec=600)  # Q1, 10:00 left
    t_late = total_time_remaining(period=4, time_remaining_period_sec=120)   # Q4, 2:00 left

    lg_early, trail_early, lead_early = compute_context_factors(
        margin=0,
        period=1,
        time_remaining_period_sec=600,
    )
    lg_late, trail_late, lead_late = compute_context_factors(
        margin=-5,
        period=4,
        time_remaining_period_sec=120,
    )

    print(f"T_early (Q1, 10:00 left): {t_early:.1f} sec")
    print(f"T_late  (Q4, 2:00 left): {t_late:.1f} sec")
    print(
        f"Early game factors -> late_game={lg_early:.3f}, trailing={trail_early:.3f}, leading={lead_early:.3f}"
    )
    print(
        f"Late game factors  -> late_game={lg_late:.3f}, trailing={trail_late:.3f}, leading={lead_late:.3f}"
    )

    _print_header("6) Apply full current context adjustments to a matchup table")
    adjusted_df = apply_context_adjustments(
        df=matchup_df,
        margin=neutral_context["margin"],
        period=neutral_context["period"],
        time_remaining_period_sec=neutral_context["time_remaining"],
        context=neutral_context,
    )

    print(
        _safe_cols(
            adjusted_df.sort_values(["PPP_CONTEXT", "NLP_CONTEXT_ADJ"], ascending=False).head(10),
            [
                "PLAY_TYPE",
                "PLAY_FAMILY",
                "PPP_CONTEXT",
                "PPP_ML_BLEND",
                "PPP_BASELINE",
                "CONTEXT_ADJ",
                "LEGACY_CONTEXT_ADJ",
                "NLP_CONTEXT_ADJ",
                "DELTA_VS_BASELINE",
                "CONTEXT_LABEL",
                "CONTEXT_PARSE_STATUS",
            ],
        ).to_string(index=False)
    )

    _print_header("7) Compare rankings: early-game vs late-game")
    early_df = rank_ml_with_context(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        margin=0,
        period=1,
        time_remaining_period_sec=600,
        k=7,
        context={
            "need": None,
            "needs": [],
            "special_situations": [],
            "preferred_play_families": [],
        },
    )

    late_df = rank_ml_with_context(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        margin=-5,
        period=4,
        time_remaining_period_sec=120,
        k=7,
        context={
            "need": "quick2",
            "needs": ["quick2", "must_score"],
            "defense_style": "switch",
            "after_timeout": True,
            "special_situations": ["after_timeout"],
            "preferred_play_families": ["ball_screen", "cut"],
            "shot_clock": 7,
            "late_clock": True,
        },
    )

    print("\nEARLY GAME (tie, Q1, 10:00 left)")
    print(
        _safe_cols(
            early_df,
            [
                "PLAY_TYPE",
                "PLAY_FAMILY",
                "PPP_CONTEXT",
                "PPP_ML_BLEND",
                "PPP_BASELINE",
                "DELTA_VS_BASELINE",
                "CONTEXT_LABEL",
            ],
        ).to_string(index=False)
    )

    print("\nLATE GAME (down 5, Q4, 2:00 left, quick2 + ATO + switch)")
    print(
        _safe_cols(
            late_df,
            [
                "PLAY_TYPE",
                "PLAY_FAMILY",
                "PPP_CONTEXT",
                "PPP_ML_BLEND",
                "PPP_BASELINE",
                "DELTA_VS_BASELINE",
                "CONTEXT_LABEL",
                "CONTEXT_PARSE_STATUS",
            ],
        ).to_string(index=False)
    )

    _print_header("8) Show ranking movement from early -> late")
    early_map: Dict[str, int] = {play: idx + 1 for idx, play in enumerate(early_df["PLAY_TYPE"].tolist())}
    late_map: Dict[str, int] = {play: idx + 1 for idx, play in enumerate(late_df["PLAY_TYPE"].tolist())}

    all_plays = sorted(set(early_map.keys()) | set(late_map.keys()))
    rows = []
    for play in all_plays:
        early_rank = early_map.get(play)
        late_rank = late_map.get(play)
        movement = None
        if early_rank is not None and late_rank is not None:
            movement = early_rank - late_rank
        rows.append(
            {
                "PLAY_TYPE": play,
                "EARLY_RANK": early_rank,
                "LATE_RANK": late_rank,
                "UP_IN_LATE_GAME": movement,
            }
        )

    movement_df = pd.DataFrame(rows).sort_values(
        by=["LATE_RANK", "EARLY_RANK"],
        na_position="last",
    )
    print(movement_df.to_string(index=False))


if __name__ == "__main__":
    main()