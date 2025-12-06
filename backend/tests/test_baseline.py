import pathlib

from baseline_recommender import BaselineRecommender

# Build an absolute path to the Synergy CSV file.
# __file__          = this test file (backend/tests/test_baseline.py)
# .parent.parent    = go up to the backend/ folder
# / "data" / ...    = then into backend/data/synergy_playtypes_2019_2025_players.csv
DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "synergy_playtypes_2019_2025_players.csv"
)


def test_rank_returns_k_rows():
    """Basic smoke test for the baseline recommender.

    This test checks two simple but important things:
    1. When we ask for K=5 recommendations, we actually get 5 rows back.
    2. The key columns we rely on ('PLAY_TYPE', 'PPP_PRED', 'PPP_GAP')
       are present in the result.

    If this test ever fails after a code change, it means our baseline
    ranking function is no longer behaving as expected.
    """
    # Create the baseline recommender using the same CSV as the main app.
    rec = BaselineRecommender(str(DATA_PATH))

    # Call the rank() function for a specific matchup.
    df = rec.rank(season="2019-20", our_team="TOR", opp_team="BOS", k=5)

    # 1) We expect exactly K rows in the result.
    assert len(df) == 5

    # 2) We expect the most important columns to exist in the DataFrame.
    for col in ["PLAY_TYPE", "PPP_PRED", "PPP_GAP"]:
        assert col in df.columns
