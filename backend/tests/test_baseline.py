import pathlib

import pandas as pd
import pytest

from domain.baseline_recommendation import BaselineRecommender

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

def test_top_pick_changes_with_the_opponent():
    """SCRUM-977: ranking by edge lets the opponent change the top pick."""
    rec = BaselineRecommender(str(DATA_PATH))
    opponents = ["BOS", "LAL", "DEN", "MIA", "GSW", "NYK"]
    tops = {
        rec.rank(season="2024-25", our_team="TOR", opp_team=opp, k=3, rank_by="edge").iloc[0]["PLAY_TYPE"]
        for opp in opponents
    }
    assert len(tops) > 1
    df = rec.rank(season="2024-25", our_team="TOR", opp_team="BOS", k=5, rank_by="edge")
    assert df["EDGE"].is_monotonic_decreasing
    assert (df["EDGE"] - (df["PPP_PRED"] - df["PPP_LEAGUE_REF"])).abs().max() < 1e-9

def test_default_order_is_predicted_ppp():
    """SCRUM-978: the baseline lists play types by predicted PPP unless edge is chosen."""
    rec = BaselineRecommender(str(DATA_PATH))
    df = rec.rank(season="2024-25", our_team="TOR", opp_team="BOS", k=5)
    assert df["PPP_PRED"].is_monotonic_decreasing


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


def test_adopt_swaps_tables_and_keeps_the_same_object(tmp_path):
    """SCRUM-490: a reload swaps tables in place so every router sees them."""
    rec = BaselineRecommender(str(DATA_PATH))
    frame = pd.read_csv(DATA_PATH)
    smaller = tmp_path / "synergy.csv"
    # Regular-season rows only (SEASON_ID 2xxxx); the file starts with playoffs.
    frame[frame["SEASON_ID"].astype(str).str.startswith("2")].head(2000).to_csv(smaller, index=False)
    fresh = BaselineRecommender(str(smaller))

    same = rec
    rec.adopt(fresh)

    assert rec is same
    assert rec.source_path == smaller
    assert rec.raw_df is fresh.raw_df
    assert rec.team_df is fresh.team_df and rec.league_df is fresh.league_df
    assert len(rec.raw_df) == 2000
    assert rec.raw_df.attrs["dataset_version"] == fresh.raw_df.attrs["dataset_version"]
