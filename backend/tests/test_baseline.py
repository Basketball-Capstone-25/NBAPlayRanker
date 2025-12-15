# backend/tests/test_baseline.py
#
# This file contains a very small but important **smoke test** for the
# BaselineRecommender.
#
# The goal isn’t to prove the model is “perfect”, but to give us an early
# warning if we accidentally:
#   - break the CSV loading / aggregation logic,
#   - change the output shape of `rank(...)`, or
#   - rename/remove the key columns the frontend relies on.
#
# If this test starts failing after a refactor, that’s a strong signal that
# something in the baseline pipeline changed and needs to be looked at.

import pathlib

from baseline_recommender import BaselineRecommender

# Build an absolute path to the Synergy CSV file.
#
# We do this instead of hard-coding a relative string path so that the test
# keeps working even if the tests are run from a different working directory.
#
# - __file__           -> this test file (backend/tests/test_baseline.py)
# - .parent            -> backend/tests/
# - .parent.parent     -> backend/
# - / "data" / ...     -> backend/data/synergy_playtypes_2019_2025_players.csv
DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "synergy_playtypes_2019_2025_players.csv"
)


def test_rank_returns_k_rows():
    """
    Basic smoke test for the baseline recommender.

    What this test is checking:

    1. If we ask the model for K = 5 recommendations, we actually get 5 rows
       back. That tells us the filtering / sorting logic isn’t accidentally
       dropping rows or returning an empty frame.

    2. The key columns we care about on the frontend are present in the
       DataFrame: 'PLAY_TYPE', 'PPP_PRED', and 'PPP_GAP'.

       - 'PLAY_TYPE' is the human-readable label we display (Spotup, PnR, etc.).
       - 'PPP_PRED' is the model’s predicted PPP for that play type in this
         matchup.
       - 'PPP_GAP' is the difference between the prediction and the opponent’s
         usual allowed PPP, which we use to explain why something is “good”.

    This isn’t an exhaustive correctness test, but it gives us a quick sanity
    check that the baseline pipeline is still producing the shape the rest of
    the system expects.
    """

    # Create the baseline recommender using the exact same CSV file that the
    # main FastAPI app will use in production.
    #
    # That way, if the CSV path changes or the schema breaks, this test will
    # fail and flag it early.
    rec = BaselineRecommender(str(DATA_PATH))

    # Call the rank() method for a concrete, realistic matchup:
    #   - season: 2019–20
    #   - our_team: Toronto (TOR)
    #   - opp_team: Boston  (BOS)
    #   - k: 5 (Top-5 recommendations)
    #
    # rank(...) is expected to return a pandas DataFrame that is already:
    #   - filtered to this season + matchup,
    #   - enriched with the predicted PPP and PPP gap,
    #   - sorted from best to worst PPP_PRED,
    #   - truncated to K rows.
    df = rec.rank(season="2019-20", our_team="TOR", opp_team="BOS", k=5)

    # 1) We expect exactly K rows in the result.
    #
    # If this assertion fails, something changed in how the recommender trims
    # results (for example, it might be returning fewer rows, or not applying
    # the Top-K logic correctly).
    assert len(df) == 5

    # 2) We expect the most important columns to exist in the DataFrame.
    #
    # If any of these columns go missing or get renamed, the frontend would
    # break (because it looks for PPP_PRED / PPP_GAP), so catching that here
    # is much nicer than discovering it during a demo.
    for col in ["PLAY_TYPE", "PPP_PRED", "PPP_GAP"]:
        assert col in df.columns
