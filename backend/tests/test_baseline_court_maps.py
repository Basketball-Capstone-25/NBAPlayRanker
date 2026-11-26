"""The Baseline diagram and its PDF must use a play's explicit court mapping."""

import csv
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import app
from domain.baseline_recommendation.playtype_zones import best_shot_zone

client = TestClient(app)
MATCHUP = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 10}


@pytest.fixture
def ranked(monkeypatch, strict_auth_by_default):
    monkeypatch.setenv("ALLOW_INSECURE_DEV_AUTH", "true")
    plays = ["Misc", "OffRebound", "PRBallHandler", "PRRollMan", "Cut"]
    rows = pd.DataFrame({
        "PLAY_TYPE": plays, "PPP_PRED": 1.05, "PPP_OFF_SHRUNK": 1.1, "PPP_DEF_SHRUNK": 1.0,
    })
    with patch("application.api_coordination.app.rank_playtypes_baseline", return_value=rows), \
         patch("application.api_coordination.export_endpoints.rank_playtypes_baseline", return_value=rows):
        yield


def test_rankings_expose_map_availability_without_dropping_plays(ranked):
    response = client.get("/rank-plays/baseline", params=MATCHUP)
    assert response.status_code == 200
    body = response.json()
    assert body["court_map_play_types"] == ["PRBallHandler", "PRRollMan", "Cut"]
    assert [row["PLAY_TYPE"] for row in body["rankings"]] == ["Misc", "OffRebound", "PRBallHandler", "PRRollMan", "Cut"]


def test_direct_csv_keeps_the_ranking_columns_and_provenance(ranked):
    page = client.get("/rank-plays/baseline", params=MATCHUP).json()["rankings"]
    download = client.get("/rank-plays/baseline.csv", params=MATCHUP)
    assert download.status_code == 200
    rows = list(csv.DictReader(StringIO(download.text)))
    assert list(rows[0]) == [*page[0], "DATASET_VERSION", "AGGREGATION_VERSION", "SEASON_PHASE"]
    assert [row["PLAY_TYPE"] for row in rows] == [row["PLAY_TYPE"] for row in page]


@pytest.mark.parametrize("play", ["Misc", "OffRebound"])
@pytest.mark.parametrize("path", ["/viz/playtype-zones", "/export/playtype-viz.pdf"])
def test_unmapped_diagram_explains_omission(ranked, play, path):
    response = client.get(path, params={**MATCHUP, "play_type": play})
    assert response.status_code == 404
    assert "no court" in response.json()["detail"].lower()
    assert "content-disposition" not in response.headers


def test_known_diagram_and_pdf_are_still_available(ranked):
    params = {**MATCHUP, "play_type": "PRRollMan"}
    response = client.get("/viz/playtype-zones", params=params)
    assert response.status_code == 200
    assert response.json()["image_base64"].startswith("iVBOR")
    pdf = client.get("/export/playtype-viz.pdf", params=params)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_context_ranker_keeps_its_fallback_label():
    assert best_shot_zone("Misc") == "At the rim"
