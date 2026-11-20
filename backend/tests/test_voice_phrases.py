"""Phrases a coach would say aloud are understood by /nlp/parse (SCRUM-626).

Speech recognition returns plain words (often numbers spelled out, no colons),
so the parser must handle them, and must report what it could not find instead
of filling it in.
"""

import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import app

client = TestClient(app)


def parse(text):
    response = client.post("/nlp/parse", json={"text": text})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("text,margin,period,time_left", [
    ("down 3, 40 seconds left, fourth quarter", -3, 4, 40),
    ("down three with 40 seconds left in the fourth quarter", -3, 4, 40),
    ("up 2 with 1:30 left in the third quarter", 2, 3, 90),
    ("tie game 30 seconds left in overtime", 0, 5, 30),
])
def test_spoken_situations_fill_all_three_fields(text, margin, period, time_left):
    body = parse(text)
    context = body["context"]
    assert body["defaulted_fields"] == [] and body["rejected_fields"] == []
    assert context["margin"] == margin
    assert context["period"] == period
    assert context["time_remaining"] == time_left


def test_missing_parts_are_reported_not_guessed():
    body = parse("down 3")
    assert body["context"]["margin"] == -3
    assert set(body["defaulted_fields"]) == {"period", "time_remaining"}
