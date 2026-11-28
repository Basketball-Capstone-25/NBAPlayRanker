import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def local_data_api_mode(strict_auth_by_default, monkeypatch):
    """Data-flow tests opt in explicitly; RBAC is verified separately."""
    monkeypatch.setenv("ALLOW_INSECURE_DEV_AUTH", "true")


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


def test_rank_baseline_api_orders_by_ppp_or_edge():
    """SCRUM-978: predicted PPP is the default order; rank_by=edge switches it."""
    params = {"season": "2019-20", "our": "TOR", "opp": "BOS", "k": 5}
    by_ppp = client.get("/rank-plays/baseline", params=params).json()
    by_edge = client.get("/rank-plays/baseline", params={**params, "rank_by": "edge"}).json()
    assert by_ppp["rank_by"] == "ppp" and by_edge["rank_by"] == "edge"
    ppp = [row["PPP_PRED"] for row in by_ppp["rankings"]]
    edge = [row["EDGE"] for row in by_edge["rankings"]]
    assert ppp == sorted(ppp, reverse=True)
    assert edge == sorted(edge, reverse=True)


def test_rank_baseline_api_rejects_unknown_rank_by():
    response = client.get(
        "/rank-plays/baseline",
        params={"season": "2019-20", "our": "TOR", "opp": "BOS", "k": 3, "rank_by": "poss"},
    )
    assert response.status_code == 422
