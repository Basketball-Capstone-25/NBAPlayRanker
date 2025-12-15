# ml_context_demo.py
#
# This is a small **driver script** for the context-aware ML recommender.
#
# It’s not part of the web API – it’s something we run from the command line to:
#   1. Sanity-check the **play-type flags** (e.g., which plays are three-point
#      oriented or “quick”).
#   2. Inspect how we compute **time remaining** and the corresponding
#      “urgency” scores.
#   3. Compare how the **context-aware ranking** changes between an early-game
#      low-urgency situation and a late-game high-urgency situation.
#
# This is very useful for defence because it gives us concrete numbers we can
# reference when we explain how context affects the ranking.

from ml_context_recommender import (
    build_ml_matchup_table,
    add_playtype_flags,
    total_time_remaining,
    compute_urgencies,
    rank_ml_with_context,
)


def main():
    # ---------------------------------------------------------
    # 1) Build the ML matchup table and inspect play-type flags
    # ---------------------------------------------------------
    #
    # build_ml_matchup_table(...) constructs the **feature table** that our
    # ML model uses for a given matchup:
    #   - It filters the underlying Synergy-derived team tables to the chosen
    #     season and teams.
    #   - It assembles one row per play type for this matchup, including
    #     features like offensive PPP, defensive PPP allowed, frequency, etc.
    #
    # In this example we use:
    #   season: "2019-20"
    #   our_team: "LAL"
    #   opp_team: "BOS"
    #
    # That gives us the raw ML input for **Lakers offense vs Celtics defense**.
    df = build_ml_matchup_table("2019-20", "LAL", "BOS")

    # add_playtype_flags(...) enriches that table with simple boolean / numeric
    # flags that capture our prior basketball knowledge about each play type,
    # for example:
    #   - THREE_PT_PRIORITY: how much this play type leans on three-point shots.
    #   - QUICK_PRIORITY: how well this play type fits “quick” late-game
    #     situations (e.g., transition, early clock actions).
    #
    # These flags are later combined with urgency scores to produce a
    # context-aware ranking.
    df_flags = add_playtype_flags(df)

    print("=== Play-type priorities (LAL offense vs BOS defense, 2019-20) ===")
    print(
        df_flags[
            [
                "PLAY_TYPE",
                "THREE_PT_PRIORITY",
                "QUICK_PRIORITY",
                "IS_3PT_ORIENTED",
                "IS_QUICK",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 2) Check time and urgency values
    # ---------------------------------------------------------
    #
    # The idea behind `total_time_remaining` is:
    #   - Convert period + time remaining in the current period into a single
    #     scalar “seconds of basketball left in regulation/OT”.
    #   - This gives us a consistent notion of "T_left" that the urgency
    #     function can use.
    #
    # Example 1: early game
    #   Q1, 10:00 left  -> lots of time.
    # Example 2: late game
    #   Q4, 2:00 left   -> very little time, so urgency should be higher.

    T_early = total_time_remaining(period=1, time_remaining_period_sec=600)   # Q1, 10:00 left
    T_late = total_time_remaining(period=4, time_remaining_period_sec=120)   # Q4, 2:00 left

    print("\n=== Time remaining checks ===")
    print(f"T_early (Q1, 10:00 left): {T_early:.1f} sec")
    print(f"T_late  (Q4, 2:00 left): {T_late:.1f} sec")

    # compute_urgencies() takes:
    #   - margin: our score minus theirs,
    #   - T_left: total time remaining (from total_time_remaining),
    # and returns two numbers:
    #   - three_urg: how urgent it is to look for 3-point oriented plays.
    #   - quick_urg: how urgent it is to look for quick-hitting plays.
    #
    # Intuition:
    #   - Early in the game with lots of time and a tie score, both urgencies
    #     should be relatively low (you don’t need to panic).
    #   - Late in the game, down multiple points with very little time left,
    #     three_urg and quick_urg should both be higher.
    three_early, quick_early = compute_urgencies(margin=0, T_left=T_early)
    three_late, quick_late = compute_urgencies(margin=-5, T_left=T_late)

    print("\n=== Urgencies ===")
    print(f"Early game, tie:       three_urg={three_early:.3f}, quick_urg={quick_early:.3f}")
    print(f"Late game, down 5:     three_urg={three_late:.3f}, quick_urg={quick_late:.3f}")

    # ---------------------------------------------------------
    # 3) Context-aware rankings for two scenarios
    # ---------------------------------------------------------
    #
    # rank_ml_with_context(...) is the main high-level function here. It:
    #   - Builds the ML matchup table for the given season / teams.
    #   - Computes T_left and urgencies based on margin + period + time_remaining.
    #   - Applies a context-aware scoring rule to each play type:
    #       CONTEXT_SCORE = f(ML_predicted_PPP, three_urg, quick_urg, play flags)
    #   - Sorts by CONTEXT_SCORE and returns the Top-K plays.
    #
    # Below we compare two very different situations:
    #
    #   A) EARLY GAME, tie score (low urgency)
    #   B) LATE GAME, down 5 with 2:00 left (high urgency)
    #
    # We keep everything else the same (same season, teams, K) so we can see how
    # the ranking shifts purely due to context.

    # A) Early-game, low urgency (Q1, 10:00 left, margin = 0).
    df_early = rank_ml_with_context(
        season="2019-20",
        our_team="LAL",
        opp_team="BOS",
        margin=0,
        period=1,
        time_remaining_period_sec=600,
        k=7,
    )

    # B) Late-game, high urgency (Q4, 2:00 left, margin = -5).
    df_late = rank_ml_with_context(
        season="2019-20",
        our_team="LAL",
        opp_team="BOS",
        margin=-5,
        period=4,
        time_remaining_period_sec=120,
        k=7,
    )

    print("\n=== EARLY GAME (low urgency) ===")
    print(df_early[["PLAY_TYPE", "PPP_PRED_ML", "CONTEXT_SCORE"]].to_string(index=False))

    print("\n=== LATE GAME (down 5, 2:00 left) ===")
    print(df_late[["PLAY_TYPE", "PPP_PRED_ML", "CONTEXT_SCORE"]].to_string(index=False))


if __name__ == "__main__":
    # Using the standard Python entry-point guard means:
    #   - main() runs when we execute this file directly:
    #         python ml_context_demo.py
    #   - but nothing runs if we import this module from somewhere else
    #     (e.g. for tests or experiments).
    main()
