"""A player's made shots against one opponent, for Context ML (SCRUM-825)."""

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.analytics_services.player_made_shots import (
    InvalidPlayerShotsFilter,
    list_players,
    parse_seasons,
    player_made_shots,
)
from application.api_coordination import player_shots_endpoints
from application.api_coordination.auth_dependency import require_auth
from application.api_coordination.shot_data_guard import require_valid_shot_data

NAMES = {"10": "Scottie Barnes", "20": "Jakob Poeltl"}


def shot(season="2024-25", team="TOR", opp="BOS", shooter="10.0", made=1, game="g1", x=41.0, y=0.0, home=True, valid=True):
    return {"SEASON_STR": season, "TEAM_ABBR": team, "OPP_ABBR": opp, "SHOOTER_ID": shooter, "MADE": made,
            "GAME_ID": game, "X": x, "Y": y, "HOME_FLAG": home, "LOCATION_VALID": valid}


SHOTS = pd.DataFrame([
    shot(),
    shot(game="g2", x=20.0, y=10.0),
    shot(made=0, x=30.0),
    shot(season="2023-24", game="g0", x=40.0, y=-3.0),
    shot(valid=False),
    shot(shooter="20", game="g2"),
    shot(shooter="99"),
    shot(opp="NYK", game="g9"),
    shot(team="BOS", opp="TOR", shooter="30"),
])


def test_only_made_shots_by_the_player_against_the_opponent_are_placed():
    result = player_made_shots(SHOTS, NAMES, "TOR", "BOS", ["2024-25"], "10")
    assert result["player"] == {"id": "10", "name": "Scottie Barnes"}
    assert result["made"] == 3 and result["attempts"] == 4
    assert result["shown"] == 2 and result["excluded_location"] == 1
    assert result["games"] == 2 and result["player_games"] == 2
    assert sorted(result["points"]) == [[20.0, 10.0], [41.0, 0.0]]


def test_several_seasons_add_up():
    result = player_made_shots(SHOTS, NAMES, "TOR", "BOS", ["2023-24", "2024-25"], "10")
    assert result["made"] == 4 and result["shown"] == 3 and result["games"] == 3


def test_away_shots_are_turned_to_the_same_basket():
    away = pd.DataFrame([shot(home=False, x=-41.0, y=2.0)])
    assert player_made_shots(away, NAMES, "TOR", "BOS", ["2024-25"], "10")["points"] == [[41.0, -2.0]]


def test_players_are_listed_by_name_with_most_makes_first():
    result = list_players(SHOTS, NAMES, "TOR", "BOS", ["2024-25"])
    assert result["players"] == [{"id": "10", "name": "Scottie Barnes", "made": 3}, {"id": "20", "name": "Jakob Poeltl", "made": 1}]
    assert result["unnamed_players"] == 1


def test_bad_filters_are_rejected():
    with pytest.raises(InvalidPlayerShotsFilter, match="Choose at least one season"):
        parse_seasons(" , ", ["2024-25"])
    with pytest.raises(InvalidPlayerShotsFilter, match="No shot data for season 2019-20"):
        parse_seasons("2019-20,2024-25", ["2024-25"])
    with pytest.raises(InvalidPlayerShotsFilter, match="Choose a player"):
        player_made_shots(SHOTS, NAMES, "TOR", "BOS", ["2024-25"], "99")
    with pytest.raises(InvalidPlayerShotsFilter, match="must be different"):
        list_players(SHOTS, NAMES, "TOR", "TOR", ["2024-25"])


def client(role="coach", names=lambda: NAMES):
    app = FastAPI()
    app.include_router(player_shots_endpoints.create_player_shots_router(lambda: SHOTS, names))
    app.dependency_overrides[require_auth] = lambda: {"role": role, "token": "fixture"}
    app.dependency_overrides[require_valid_shot_data] = lambda: None
    return TestClient(app)


def test_api_returns_players_and_points_for_coaches():
    api = client()
    listed = api.get("/viz/player-made-shots/players", params={"our": "tor", "opp": "bos", "seasons": "2024-25"})
    assert listed.status_code == 200 and listed.json()["players"][0]["name"] == "Scottie Barnes"
    shots = api.get("/viz/player-made-shots", params={"our": "TOR", "opp": "BOS", "seasons": "2023-24,2024-25", "player": "10"})
    assert shots.status_code == 200 and shots.json()["made"] == 4


def test_api_explains_bad_seasons_and_missing_names():
    assert client().get("/viz/player-made-shots", params={"our": "TOR", "opp": "BOS", "seasons": "1999-00", "player": "10"}).status_code == 400

    def missing():
        raise player_shots_endpoints.PlayerShotsUnavailable(player_shots_endpoints.BUILD_HINT)

    response = client(names=missing).get("/viz/player-made-shots/players", params={"our": "TOR", "opp": "BOS", "seasons": "2024-25"})
    assert response.status_code == 503 and "build_shooter_names.py" in response.json()["detail"]


def test_roles_without_shot_tools_are_refused():
    assert client(role=None).get("/viz/player-made-shots/players", params={"our": "TOR", "opp": "BOS", "seasons": "2024-25"}).status_code == 403


def test_api_lists_the_seasons_with_shot_data():
    assert client().get("/viz/player-made-shots/seasons").json() == {"seasons": ["2023-24", "2024-25"]}
