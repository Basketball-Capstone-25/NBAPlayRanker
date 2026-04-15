from fastapi.testclient import TestClient

from application.api_coordination.app import app


client = TestClient(app)


def test_rank_baseline_api_returns_top_k_rows():
    """Ranked list via API returns expected metadata and top-k rows."""
    response = client.get(
        "/rank-plays/baseline",
        params={"season": "2019-20", "our": "TOR", "opp": "BOS", "k": 3},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["season"] == "2019-20"
    assert payload["our_team"] == "TOR"
    assert payload["opp_team"] == "BOS"
    assert payload["k"] == 3
    assert len(payload["rankings"]) == 3

    first_row = payload["rankings"][0]
    for key in ["PLAY_TYPE", "PPP_PRED", "PPP_GAP"]:
        assert key in first_row


def test_rank_baseline_api_rejects_same_team_matchup():
    """API should return 400 when our team and opponent are the same."""
    response = client.get(
        "/rank-plays/baseline",
        params={"season": "2019-20", "our": "TOR", "opp": "TOR", "k": 5},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Our team and opponent must be different."


def test_rank_baseline_api_rejects_invalid_k():
    """FastAPI query validation should reject k outside [1, 10]."""
    response = client.get(
        "/rank-plays/baseline",
        params={"season": "2019-20", "our": "TOR", "opp": "BOS", "k": 0},
    )

    assert response.status_code == 422
