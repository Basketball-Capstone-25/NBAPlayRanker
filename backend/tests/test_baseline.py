import pathlib

import pytest

from baseline_recommender import BaselineRecommender

#
#
# - .parent            -> backend/tests/
# - .parent.parent     -> backend/
DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "synergy_playtypes_2019_2025_players.csv"
)

def test_rank_returns_k_rows():
    """Unit tests for BaselineRecommender."""

    # main FastAPI app will use in production.
    #
    # fail and flag it early.
    rec = BaselineRecommender(str(DATA_PATH))

    #   - season: 2019–20
    #   - our_team: Toronto (TOR)
    #   - opp_team: Boston  (BOS)
    #   - k: 5 (Top-5 recommendations)
    #
    #   - filtered to this season + matchup,
    #   - sorted from best to worst PPP_PRED,
    #   - truncated to K rows.
    df = rec.rank(season="2019-20", our_team="TOR", opp_team="BOS", k=5)

    #
    # the Top-K logic correctly).
    assert len(df) == 5

    #
    for col in ["PLAY_TYPE", "PPP_PRED", "PPP_GAP"]:
        assert col in df.columns

def test_rank_ppp_pred_non_negative():
    rec = BaselineRecommender(str(DATA_PATH))
    df = rec.rank(season="2019-20", our_team="TOR", opp_team="BOS", k=5)
    assert (df["PPP_PRED"] >= 0).all()

def test_rank_k_exceeds_available():
    rec = BaselineRecommender(str(DATA_PATH))
    df = rec.rank(season="2019-20", our_team="TOR", opp_team="BOS", k=10)
    assert len(df) > 0
    assert len(df) <= 10

def test_different_teams_produce_different_rankings():
    rec = BaselineRecommender(str(DATA_PATH))
    df1 = rec.rank(season="2019-20", our_team="LAL", opp_team="BOS", k=5)
    df2 = rec.rank(season="2019-20", our_team="MIA", opp_team="BOS", k=5)
    assert not df1["PPP_PRED"].values.tolist() == df2["PPP_PRED"].values.tolist()

def test_shot_baseline_rank_returns_dict():
    from domain.baseline_recommendation.shot_baseline_recommender import ShotBaselineRecommender
    agg = pathlib.Path(__file__).resolve().parent.parent / "data" / "pbp" / "shots_agg.parquet"
    league = pathlib.Path(__file__).resolve().parent.parent / "data" / "pbp" / "shots_agg_league.parquet"
    if not agg.exists() or not league.exists():
        pytest.skip("shot parquet files not available")
    rec = ShotBaselineRecommender(agg_path=agg, league_path=league)
    result = rec.rank(season=rec.available_seasons[0], our_team=rec.available_teams[0], opp_team=rec.available_teams[1], k=3)
    assert isinstance(result, dict)
    assert "by_shot_type" in result or "by_zone" in result or len(result) > 0
