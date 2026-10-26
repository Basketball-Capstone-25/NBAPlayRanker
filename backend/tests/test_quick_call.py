"""Timeout Quick-Call Mode (SCRUM-628): top 3, shot zones, fallback and access."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.app import app
from application.recommendation_services.quick_call_service import (
    BASELINE_NOTICE,
    QuickCallRequest,
    QuickCallService,
)
from domain.baseline_recommendation.playtype_zones import best_shot_zone
from domain.context_ml_recommendation.quick_call import build_quick_call

client = TestClient(app)
PARAMS = {"season": "2019-20", "our": "TOR", "opp": "BOS", "margin": -3, "period": 4, "time_remaining": 28}
REQUEST = QuickCallRequest("2019-20", "TOR", "BOS", -3.0, 4, 28.0)

CONTEXT_ROWS = pd.DataFrame(
    [
        {"PLAY_TYPE": "PRRollMan", "PPP_CONTEXT": 1.181, "DELTA_VS_BASELINE": 0.042, "CONTEXT_LABEL": "Late & trailing | must_score"},
        {"PLAY_TYPE": "Spotup", "PPP_CONTEXT": 1.104, "DELTA_VS_BASELINE": 0.031, "CONTEXT_LABEL": "Late & trailing | must_score"},
        {"PLAY_TYPE": "Isolation", "PPP_CONTEXT": 0.951, "DELTA_VS_BASELINE": -0.01, "CONTEXT_LABEL": ""},
        {"PLAY_TYPE": "Postup", "PPP_CONTEXT": 0.94, "DELTA_VS_BASELINE": 0.0, "CONTEXT_LABEL": ""},
    ]
)
BASELINE_ROWS = pd.DataFrame(
    [
        {"PLAY_TYPE": "Cut", "PPP_PRED": 1.25, "PPP_OFF_SHRUNK": 1.31, "PPP_DEF_SHRUNK": 1.2},
        {"PLAY_TYPE": "Transition", "PPP_PRED": 1.12, "PPP_OFF_SHRUNK": 1.15, "PPP_DEF_SHRUNK": 1.1},
        {"PLAY_TYPE": "Handoff", "PPP_PRED": 0.97, "PPP_OFF_SHRUNK": 0.98, "PPP_DEF_SHRUNK": 0.95},
    ]
)


def test_shot_zone_is_the_highest_weight_zone_for_each_play_type():
    assert best_shot_zone("PRRollMan") == "At the rim"
    assert best_shot_zone("Spotup") == "Corner 3"
    assert best_shot_zone("Postup") == "Paint"
    assert best_shot_zone("OffScreen") == "Above-the-break 3"
    assert best_shot_zone("Unknown play") == "At the rim"


def test_quick_call_keeps_only_three_plays_with_zone_and_one_line_reason():
    plays = build_quick_call(CONTEXT_ROWS.to_dict(orient="records"), "context")

    assert [p.rank for p in plays] == [1, 2, 3]
    assert [p.play_type for p in plays] == ["PRRollMan", "Spotup", "Isolation"]
    assert plays[0].shot_zone == "At the rim"
    assert plays[0].reason == "+0.04 PPP vs season baseline (late & trailing)"
    assert plays[2].reason == "-0.01 PPP vs season baseline"
    assert all("\n" not in p.reason and len(p.reason) <= 110 for p in plays)


def test_rows_without_a_usable_ppp_are_skipped():
    rows = [{"PLAY_TYPE": "Cut", "PPP_CONTEXT": float("nan")}, {"PLAY_TYPE": "Spotup", "PPP_CONTEXT": 1.0}]
    plays = build_quick_call(rows, "context")
    assert [(p.rank, p.play_type) for p in plays] == [(1, "Spotup")]


def test_service_uses_context_model_when_it_answers():
    service = QuickCallService(lambda _: CONTEXT_ROWS, lambda _: pytest.fail("baseline must not run"))
    result = service.recommend(REQUEST)
    assert result.source == "context"
    assert result.notice == ""


@pytest.mark.parametrize("failure", [ValueError("no model"), FileNotFoundError("missing artifact"), RuntimeError("boom")])
def test_service_falls_back_to_baseline_and_says_so(failure):
    def broken(_):
        raise failure

    result = QuickCallService(broken, lambda _: BASELINE_ROWS).recommend(REQUEST)

    assert result.source == "baseline"
    assert result.notice == BASELINE_NOTICE
    assert [p.play_type for p in result.plays] == ["Cut", "Transition", "Handoff"]
    assert result.plays[0].reason == "Our offense 1.31 PPP vs their defense allowing 1.20"


def test_service_falls_back_when_context_returns_nothing():
    result = QuickCallService(lambda _: pd.DataFrame(), lambda _: BASELINE_ROWS).recommend(REQUEST)
    assert result.source == "baseline"


@pytest.fixture
def dev_auth(monkeypatch):
    monkeypatch.setenv("ALLOW_INSECURE_DEV_AUTH", "true")


def test_api_returns_three_context_plays_with_situation(dev_auth):
    response = client.get("/rank-plays/quick-call", params=PARAMS)

    assert response.status_code == 200
    body = response.json()
    assert body["source"] in {"context", "baseline"}
    assert len(body["plays"]) == 3
    assert {"rank", "play_type", "ppp", "shot_zone", "reason"} <= set(body["plays"][0])
    assert body["situation"] == {
        "season": "2019-20", "our_team": "TOR", "opp_team": "BOS", "margin": -3.0, "period": 4, "time_remaining": 28.0,
    }
    assert isinstance(body["elapsed_ms"], int)


def test_api_labels_baseline_fallback_when_context_model_fails(dev_auth, monkeypatch):
    def broken(_):
        raise FileNotFoundError("model artifact missing")

    monkeypatch.setattr(app_module.quick_call_service, "_context_ranker", broken)
    body = client.get("/rank-plays/quick-call", params=PARAMS).json()

    assert body["source"] == "baseline"
    assert body["notice"] == BASELINE_NOTICE
    assert len(body["plays"]) == 3


@pytest.mark.parametrize(
    "override, status",
    [
        ({"margin": 41}, 400),
        ({"period": 5, "time_remaining": 301}, 400),
        ({"time_remaining": 721}, 422),
        ({"period": 6}, 422),
        ({"opp": "TOR"}, 400),
        ({"season": "1999-00"}, 400),
    ],
)
def test_api_rejects_impossible_situations(dev_auth, override, status):
    assert client.get("/rank-plays/quick-call", params={**PARAMS, **override}).status_code == status


def test_signed_out_user_cannot_open_quick_call():
    assert client.get("/rank-plays/quick-call", params=PARAMS).status_code in {401, 403}


def test_analyst_is_refused_quick_call():
    service = "application.access_control_services.access_control_service"
    with patch(f"{service}.decode_supabase_jwt", return_value={"sub": "user-1", "exp": 9999999999}), patch(
        f"{service}.get_profile_role", return_value="analyst"
    ):
        response = client.get("/rank-plays/quick-call", params=PARAMS, headers={"Authorization": "Bearer t"})
    assert response.status_code == 403
