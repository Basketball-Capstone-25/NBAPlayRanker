from ml_context_recommender import (
    build_ml_matchup_table,
    add_playtype_flags,
    total_time_remaining,
    compute_urgencies,
    rank_ml_with_context,
)


def main():
    # 1) Build ML matchup table (no context yet)
    df = build_ml_matchup_table("2019-20", "LAL", "BOS")
    df_flags = add_playtype_flags(df)

    print("=== Play-type priorities (LAL offense vs BOS defense, 2019-20) ===")
    print(
        df_flags[
            ["PLAY_TYPE", "THREE_PT_PRIORITY", "QUICK_PRIORITY", "IS_3PT_ORIENTED", "IS_QUICK"]
        ].to_string(index=False)
    )

    # 2) Check time and urgency values
    T_early = total_time_remaining(period=1, time_remaining_period_sec=600)   # Q1, 10:00 left
    T_late = total_time_remaining(period=4, time_remaining_period_sec=120)   # Q4, 2:00 left

    print("\n=== Time remaining checks ===")
    print(f"T_early (Q1, 10:00 left): {T_early:.1f} sec")
    print(f"T_late  (Q4, 2:00 left): {T_late:.1f} sec")

    three_early, quick_early = compute_urgencies(margin=0, T_left=T_early)
    three_late, quick_late = compute_urgencies(margin=-5, T_left=T_late)

    print("\n=== Urgencies ===")
    print(f"Early game, tie:       three_urg={three_early:.3f}, quick_urg={quick_early:.3f}")
    print(f"Late game, down 5:     three_urg={three_late:.3f}, quick_urg={quick_late:.3f}")

    # 3) Context rankings for two scenarios
    df_early = rank_ml_with_context(
        season="2019-20",
        our_team="LAL",
        opp_team="BOS",
        margin=0,
        period=1,
        time_remaining_period_sec=600,
        k=7,
    )

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
    main()
