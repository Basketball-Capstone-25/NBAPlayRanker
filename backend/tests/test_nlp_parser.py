from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


# Make backend imports work when tests are run from repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nlp_endpoints import router  # noqa: E402


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_nlp_health_endpoint():
    client = make_client()

    res = client.get("/nlp/health")
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "ok"
    assert data["router"] == "nlp"
    assert data["ready"] is True


def test_parse_endpoint_returns_advanced_context_and_context_ml_params():
    client = make_client()

    payload = {
        "text": "Down 3 with 0:28 left in Q4, need a quick 2, they're switching everything after timeout.",
        "defaults": {
            "shot_clock": 9,
        },
    }

    res = client.post("/nlp/parse", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()

    assert data["raw_text"] == payload["text"]
    assert data["confidence"] > 0.5
    assert "context" in data
    assert "context_ml_params" in data

    ctx = data["context"]
    ml = data["context_ml_params"]

    assert ctx["period"] == 4
    assert ctx["time_remaining"] == 28.0
    assert ctx["margin"] == -3.0
    assert ctx["need"] == "quick2"
    assert ctx["quick2"] is True
    assert ctx["defense_style"] == "switch"
    assert ctx["vs_switching"] is True
    assert ctx["after_timeout"] is True
    assert ctx["shot_clock"] == 9.0

    assert ml["period"] == 4
    assert ml["margin"] == -3.0
    assert ml["time_remaining"] == 28.0
    assert ml["need"] == "quick2"
    assert ml["quick2"] is True
    assert ml["defense_style"] == "switch"
    assert ml["vs_switching"] is True
    assert ml["after_timeout"] is True


def test_parse_endpoint_422_on_blank_text():
    client = make_client()

    res = client.post("/nlp/parse", json={"text": "   "})
    assert res.status_code == 422


def test_explain_endpoint_with_structured_context_and_rankings():
    client = make_client()

    payload = {
        "context": {
            "period": 4,
            "time_remaining": 28,
            "margin": -3,
            "need": "quick2",
            "quick2": True,
            "defense_style": "switch",
            "vs_switching": True,
            "after_timeout": True,
            "preferred_play_families": ["ball_screen"],
            "context_brief": "Q4 • 0:28 left • down 3 • quick 2 • switching • after timeout",
            "objective_summary": "Priority is generating a fast, efficient 2-point look and staying ahead of the clock.",
        },
        "ranked_context": {
            "rankings": [
                {
                    "PLAY_TYPE": "P&R Ball Handler",
                    "PPP_CONTEXT": 1.08,
                    "PPP_ML_BLEND": 1.06,
                    "PPP_BASELINE": 1.00,
                    "DELTA_VS_BASELINE": 0.08,
                    "RATIONALE": "Best blend of pace and efficiency.",
                },
                {
                    "PLAY_TYPE": "Post Up",
                    "PPP_CONTEXT": 1.02,
                    "PPP_BASELINE": 1.00,
                },
            ]
        },
        "ranked_baseline": {
            "rankings": [
                {
                    "PLAY_TYPE": "P&R Ball Handler",
                    "PPP_PRED": 1.00,
                },
                {
                    "PLAY_TYPE": "Post Up",
                    "PPP_PRED": 1.00,
                },
            ]
        },
        "top_k": 2,
        "parser_warnings": ["carry parser warning"],
        "clarifying_questions": ["What is the exact shot clock?"],
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()

    assert data["context_summary"] == payload["context"]["context_brief"]
    assert data["overall_summary"] == payload["context"]["objective_summary"]
    assert data["mode"] == "context-ml"
    assert isinstance(data["plays"], list)
    assert isinstance(data["explanation"], list)
    assert len(data["plays"]) == 2
    assert data["plays"][0]["play_name"] == "P&R Ball Handler"
    assert data["plays"][0]["rank"] == 1
    assert data["parser_warnings"] == ["carry parser warning"]
    assert data["clarifying_questions"] == ["What is the exact shot clock?"]


def test_explain_endpoint_can_parse_text_inline_and_merge_with_context():
    client = make_client()

    payload = {
        "text": "ATO, down 3, 18 on the shot clock, 0:32 in OT, need a clean 3.",
        "defaults": {
            "margin": -2,
        },
        "context": {
            "preferred_play_families": ["ball_screen"],
        },
        "ranked_context": {
            "rankings": [
                {
                    "PLAY_TYPE": "P&R Ball Handler",
                    "PPP_CONTEXT": 1.07,
                    "PPP_BASELINE": 1.00,
                }
            ]
        },
        "top_k": 1,
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()

    assert data["context_summary"]
    assert data["overall_summary"]
    assert len(data["plays"]) == 1
    assert data["plays"][0]["play_name"] == "P&R Ball Handler"
    assert any("warning" in x.lower() or isinstance(x, str) for x in data["parser_warnings"])


def test_explain_endpoint_supports_legacy_rankings_and_mode_payload():
    client = make_client()

    payload = {
        "mode": "baseline",
        "context": {
            "period": 4,
            "time_remaining": 45,
            "margin": 0,
        },
        "rankings": [
            {
                "PLAY_TYPE": "Spotup",
                "PPP_PRED": 1.02,
            }
        ],
        "top_n": 1,
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()

    assert data["mode"] == "baseline"
    assert len(data["plays"]) == 1
    assert data["plays"][0]["play_name"] == "Spotup"
    assert data["explanation"][0]["play_name"] == "Spotup"


def test_explain_endpoint_includes_shotplan_explanation_when_provided():
    client = make_client()

    payload = {
        "context": {
            "period": 4,
            "time_remaining": 18,
            "margin": -3,
            "need3": True,
        },
        "ranked_context": {
            "rankings": [
                {
                    "PLAY_TYPE": "Post Up",
                    "PPP_CONTEXT": 1.03,
                }
            ]
        },
        "shotplan": {
            "shot_type": "At Rim",
            "zone": "Restricted Area",
            "expected_ppp": 1.18,
            "rationale": "Best expected value at the rim.",
        },
        "top_k": 1,
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()

    assert data["shotplan_explanation"] is not None
    assert "summary" in data["shotplan_explanation"]
    assert "At Rim" in data["shotplan_explanation"]["summary"]


def test_explain_endpoint_422_when_missing_context_and_text():
    client = make_client()

    payload = {
        "ranked_context": {
            "rankings": [
                {
                    "PLAY_TYPE": "Spotup",
                    "PPP_CONTEXT": 1.02,
                }
            ]
        }
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 422


def test_explain_endpoint_422_when_rankings_missing():
    client = make_client()

    payload = {
        "context": {
            "period": 4,
            "time_remaining": 30,
            "margin": -2,
        }
    }

    res = client.post("/nlp/explain", json=payload)
    assert res.status_code == 422


def test_full_parse_then_explain_gameplan_style_flow():
    client = make_client()

    parse_res = client.post(
        "/nlp/parse",
        json={
            "text": "Down 3 with 0:28 left in Q4, need a quick 2, they're switching everything after timeout.",
            "defaults": {
                "period": 4,
                "time_remaining": 40,
                "margin": -3,
            },
        },
    )
    assert parse_res.status_code == 200, parse_res.text

    parse_data = parse_res.json()
    ctx = parse_data["context"]

    explain_res = client.post(
        "/nlp/explain",
        json={
            "context": ctx,
            "ranked_context": {
                "rankings": [
                    {
                        "PLAY_TYPE": "P&R Ball Handler",
                        "PPP_CONTEXT": 1.08,
                        "PPP_ML_BLEND": 1.06,
                        "PPP_BASELINE": 1.00,
                        "DELTA_VS_BASELINE": 0.08,
                        "RATIONALE": "Best blend of pace and efficiency.",
                    },
                    {
                        "PLAY_TYPE": "Spotup",
                        "PPP_CONTEXT": 1.01,
                        "PPP_BASELINE": 0.99,
                    },
                ]
            },
            "ranked_baseline": {
                "rankings": [
                    {"PLAY_TYPE": "P&R Ball Handler", "PPP_PRED": 1.00},
                    {"PLAY_TYPE": "Spotup", "PPP_PRED": 0.99},
                ]
            },
            "parser_warnings": parse_data.get("warnings", []),
            "clarifying_questions": parse_data.get("clarifying_questions", []),
            "top_k": 2,
        },
    )
    assert explain_res.status_code == 200, explain_res.text

    explain_data = explain_res.json()

    assert explain_data["context_summary"]
    assert explain_data["overall_summary"]
    assert len(explain_data["plays"]) == 2
    assert explain_data["plays"][0]["play_name"] == "P&R Ball Handler"
    assert any(
        "Context-adjusted efficiency" in bullet
        for bullet in explain_data["plays"][0]["evidence"]
    )
