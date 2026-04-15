from fastapi.testclient import TestClient

from application.api_coordination.app import app


client = TestClient(app)


def test_access_workspace_returns_table_with_meta():
    """Access workspace returns rows plus summary metadata."""
    response = client.get(
        "/data/team-playtypes",
        params={"season": "2019-20"},
    )

    assert response.status_code == 200

    payload = response.json()
    for key in ["season", "total_rows", "returned_rows", "rows"]:
        assert key in payload

    assert payload["season"] == "2019-20"
    assert isinstance(payload["rows"], list)
    assert payload["returned_rows"] == len(payload["rows"])


def test_access_workspace_applies_team_and_side_filters():
    """Team and side filters narrow rows to the requested slice."""
    response = client.get(
        "/data/team-playtypes",
        params={
            "season": "2019-20",
            "team": "TOR",
            "side": "offense",
            "limit": 100,
        },
    )

    assert response.status_code == 200
    rows = response.json()["rows"]

    for row in rows:
        assert row["SEASON"] == "2019-20"
        assert row["TEAM_ABBREVIATION"] == "TOR"
        assert row["SIDE"] == "offense"


def test_access_workspace_respects_limit_and_reports_counts():
    """Limit is honored and metadata counts stay consistent."""
    limit = 7
    response = client.get(
        "/data/team-playtypes",
        params={"season": "2019-20", "limit": limit},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["returned_rows"] <= limit
    assert payload["returned_rows"] == len(payload["rows"])
    assert payload["total_rows"] >= payload["returned_rows"]
