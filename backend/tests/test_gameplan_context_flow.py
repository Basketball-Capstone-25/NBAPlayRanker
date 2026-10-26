"""Parser → HTTP contract → real context adjustments, using a fixed matchup table."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import app
from domain.context_ml_recommendation import ml_context_recommender as ranker

client = TestClient(app)
MATCHUP = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 3}


@pytest.fixture(autouse=True)
def fixed_matchup(strict_auth_by_default, monkeypatch):
    monkeypatch.setenv("ALLOW_INSECURE_DEV_AUTH", "true")
    # The endpoint and context adjustment code are real. Keep the learned
    # season estimates fixed so condition effects can be checked exactly.
    monkeypatch.setattr(ranker, "build_ml_matchup_table", lambda **kwargs: pd.DataFrame({
        "PLAY_TYPE": ["Spotup", "Postup", "Cut"],
        "PLAY_FAMILY": ["spot_up", "post_up", "cut"],
        "PPP_ML_BLEND": [1.0, 1.0, 1.0], "PPP_BASELINE": [1.0, 1.0, 1.0],
        "EFFECTIVE_W_OFF": [0.7, 0.7, 0.7], "EFFECTIVE_W_DEF": [0.3, 0.3, 0.3],
        "EFG_PCT_OFF": [0.5, 0.5, 0.5], "TOV_POSS_PCT_OFF": [0.1, 0.1, 0.1],
    }))


def request_rank(context, coaching=None):
    params = {**MATCHUP, **{key: context[key] for key in ["margin", "period", "time_remaining"]}}
    if coaching is not None:
        params["coaching"] = json.dumps(coaching)
    return client.get("/rank-plays/context-ml", params=params)


def test_first_and_second_requests_use_the_same_parsed_context():
    parsed = client.post("/nlp/parse", json={"text": "Q4 down 5 with 0:28 left", "defaults": {"margin": -2, "period": 4, "time_remaining": 180}})
    assert parsed.status_code == 200
    context = parsed.json()["context"]
    first = request_rank(context)
    second = request_rank(context)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["margin"] == -5
    assert first.json()["period"] == 4
    assert first.json()["time_remaining_period_sec"] == 28
    assert all("Used score margin -5, period 4 and 28 seconds" in row["RATIONALE"] for row in first.json()["rankings"])


@pytest.mark.parametrize("coaching, play, detail", [
    ({"need3": True}, "Spotup", "need=need3"),
    ({"vs_switching": True}, "Cut", "defense=switch"),
])
def test_supported_conditions_change_actual_scores(coaching, play, detail):
    situation = {"margin": -3, "period": 4, "time_remaining": 28}
    normal = request_rank(situation).json()
    changed = request_rank(situation, coaching)
    assert changed.status_code == 200
    before = {row["PLAY_TYPE"]: row for row in normal["rankings"]}
    after = {row["PLAY_TYPE"]: row for row in changed.json()["rankings"]}
    assert after[play]["PPP_CONTEXT"] > before[play]["PPP_CONTEXT"]
    assert detail in after[play]["RATIONALE"]


def test_ot_survives_parser_to_ranker_and_rejects_an_invalid_clock():
    parsed = client.post("/nlp/parse", json={"text": "OT down 2 with 0:28 left"})
    context = parsed.json()["context"]
    result = request_rank(context)
    assert result.status_code == 200
    assert context["period"] == result.json()["period"] == 5
    assert result.json()["time_remaining_period_sec"] == 28
    assert request_rank({**context, "time_remaining": 301}).status_code == 422
    adjusted = client.post("/nlp/parse", json={"text": "OT down 2 with 6:00 left"}).json()
    assert adjusted["context"]["period"] == 5
    assert adjusted["context"]["time_remaining"] == 300
    assert any("300 seconds" in warning for warning in adjusted["warnings"])


@pytest.mark.parametrize("coaching", [{"need3": "yes"}, {"defense_style": "unknown"}, {"period": 1}, {"arbitrary": True}, {"shot_clock": 25}])
def test_invalid_conditions_are_rejected_instead_of_silently_ignored(coaching):
    assert request_rank({"margin": -3, "period": 4, "time_remaining": 28}, coaching).status_code == 422


def test_baseline_rationale_discloses_missing_context_for_every_row():
    result = client.get("/rank-plays/baseline", params={**MATCHUP, "k": 5})
    assert result.status_code == 200
    rows = result.json()["rankings"]
    assert len(rows) == 5
    assert all("score margin, period and clock do not affect this baseline ranking" in row["RATIONALE"] for row in rows)
    assert all("Reliability" in row["RATIONALE"] for row in rows)


@pytest.mark.parametrize("source, family", [("Spotup", "spot_up"), ("OffScreen", "off_screen"), ("PRBallHandler", "ball_screen"), ("PRRollMan", "ball_screen")])
def test_source_play_names_have_a_supported_context_family(source, family):
    assert ranker._play_family(source) == family
