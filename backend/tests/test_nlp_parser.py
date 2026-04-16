from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


# -- fallback / error-handling tests --

def test_unparseable_prompt_uses_ui_defaults():
    """Vague prompt should fall back to whatever defaults the frontend sends."""
    client = make_client()

    res = client.post("/nlp/parse", json={
        "text": "Just win the game",
        "defaults": {"period": 4, "time_remaining": 180, "margin": 0},
    })
    assert res.status_code == 200

    ctx = res.json()["context"]
    # The vague text has nothing to extract, so defaults must fill in.
    assert ctx["period"] == 4
    assert ctx["time_remaining"] == 180.0
    assert ctx["margin"] == 0.0


def test_unparseable_prompt_low_confidence_and_questions():
    """Vague input → lower confidence + clarifying questions."""
    client = make_client()

    res = client.post("/nlp/parse", json={
        "text": "Just win the game",
        "defaults": {"period": 4, "time_remaining": 180, "margin": 0},
    })
    data = res.json()

    # defaults push confidence up a bit but it should still be
    # lower than a fully-detailed prompt (~0.99)
    assert data["confidence"] < 0.95, "confidence too high for a vague prompt"
    assert len(data.get("clarifying_questions", [])) > 0, "should ask follow-up questions"


def test_unparseable_prompt_without_defaults_still_200():
    """No defaults + vague text should still return 200, not crash."""
    client = make_client()

    res = client.post("/nlp/parse", json={"text": "Let's go team"})
    assert res.status_code == 200

    ctx = res.json()["context"]
    # Without defaults, fields remain None — that's acceptable.
    assert ctx.get("period") is None or isinstance(ctx["period"], (int, float))


# -- extraction accuracy tests (margin, period, time, defense) --

MARGIN_CASES = [
    ("down 5",             -5.0),
    ("Down 3",             -3.0),
    ("trailing by 10",     -10.0),
    ("behind by 7",        -7.0),
    ("leading by 3",        3.0),
    ("up 8",                8.0),
    ("ahead by 12",         12.0),
    ("tied",                0.0),
    ("score is even",       0.0),
]


@pytest.mark.parametrize("phrase,expected_margin", MARGIN_CASES,
                         ids=[c[0] for c in MARGIN_CASES])
def test_margin_extraction_accuracy(phrase, expected_margin):
    """'down 5' → -5, 'up 3' → +3, etc."""
    client = make_client()

    res = client.post("/nlp/parse", json={"text": phrase})
    assert res.status_code == 200

    ctx = res.json()["context"]
    assert ctx["margin"] == expected_margin, (
        f"'{phrase}' should parse to margin={expected_margin}, got {ctx['margin']}"
    )


PERIOD_CASES = [
    ("Q1 situation",   1),
    ("in Q2",          2),
    ("third quarter",  3),
    ("Q4 crunch time", 4),
    ("in overtime",    5),
]


@pytest.mark.parametrize("phrase,expected_period", PERIOD_CASES,
                         ids=[c[0] for c in PERIOD_CASES])
def test_period_extraction_accuracy(phrase, expected_period):
    """Check that Q1/Q2/.../OT get parsed to the right period number."""
    client = make_client()

    res = client.post("/nlp/parse", json={"text": phrase})
    assert res.status_code == 200

    ctx = res.json()["context"]
    assert ctx["period"] == expected_period, (
        f"'{phrase}' should parse to period={expected_period}, got {ctx['period']}"
    )


TIME_CASES = [
    ("0:28 left",          28.0),
    ("2:15 remaining",     135.0),
    ("45 seconds left",    45.0),
    ("5 minutes remaining", 300.0),
]


@pytest.mark.parametrize("phrase,expected_seconds", TIME_CASES,
                         ids=[c[0] for c in TIME_CASES])
def test_time_extraction_accuracy(phrase, expected_seconds):
    """'0:28 left' → 28s, '5 minutes remaining' → 300s, etc."""
    client = make_client()

    res = client.post("/nlp/parse", json={"text": phrase})
    assert res.status_code == 200

    ctx = res.json()["context"]
    assert ctx["time_remaining"] == expected_seconds, (
        f"'{phrase}' should parse to time_remaining={expected_seconds}, "
        f"got {ctx['time_remaining']}"
    )


DEFENSE_CASES = [
    ("they're switching everything", "switch"),
    ("against a zone defense",       "generic_zone"),
]


@pytest.mark.parametrize("phrase,expected_defense", DEFENSE_CASES,
                         ids=[c[0] for c in DEFENSE_CASES])
def test_defense_style_extraction(phrase, expected_defense):
    """Make sure switching/zone defense gets picked up."""
    client = make_client()

    res = client.post("/nlp/parse", json={"text": phrase})
    assert res.status_code == 200

    ctx = res.json()["context"]
    assert ctx["defense_style"] == expected_defense, (
        f"'{phrase}' should parse to defense_style='{expected_defense}', "
        f"got '{ctx['defense_style']}'"
    )


# -- determinism & template-based explanations --

def test_explanations_are_deterministic():
    """Same request twice → same output. No randomness in the templates."""
    client = make_client()

    payload = {
        "mode": "baseline",
        "context": {"period": 4, "time_remaining": 30, "margin": -3},
        "rankings": [
            {"PLAY_TYPE": "Transition", "PPP_PRED": 1.10},
            {"PLAY_TYPE": "Spotup", "PPP_PRED": 0.98},
        ],
        "top_n": 2,
    }

    res1 = client.post("/nlp/explain", json=payload).json()
    res2 = client.post("/nlp/explain", json=payload).json()

    assert res1["plays"] == res2["plays"], "got different results on same input??"
    assert res1["overall_summary"] == res2["overall_summary"]


def test_explanation_evidence_contains_real_metrics():
    """Evidence should include the actual PPP value we passed in."""
    client = make_client()

    payload = {
        "mode": "baseline",
        "context": {"period": 4, "time_remaining": 30, "margin": -3},
        "rankings": [
            {"PLAY_TYPE": "Transition", "PPP_PRED": 1.10},
        ],
        "top_n": 1,
    }

    res = client.post("/nlp/explain", json=payload)
    data = res.json()

    play = data["plays"][0]
    evidence_text = " ".join(play["evidence"])
    # make sure it actually uses the PPP we gave it
    assert "1.10" in evidence_text or "1.1" in evidence_text, (
        f"Expected PPP 1.10 somewhere in evidence, got: {play['evidence']}"
    )
