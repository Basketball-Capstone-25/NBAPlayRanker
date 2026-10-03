"""Hand-computed arithmetic, scope, export parity, and analyst access checks."""

import csv
import hashlib
import io

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.api_coordination.auth_dependency import require_auth
from application.api_coordination.topk_uplift_endpoints import create_topk_uplift_router
from domain.statistical_analysis.topk_uplift import compute_topk_uplift, UpliftDataUnavailable


@pytest.fixture
def source_tables():
    team = pd.DataFrame([
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "A", "PTS": 100, "POSS": 100},
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "B", "PTS": 600, "POSS": 300},
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "TOR", "SIDE": "defense", "PLAY_TYPE": "A", "PTS": 10000, "POSS": 1},
        {"SEASON": "2024-25", "TEAM_ABBREVIATION": "BOS", "SIDE": "offense", "PLAY_TYPE": "A", "PTS": 10000, "POSS": 1},
        {"SEASON": "2023-24", "TEAM_ABBREVIATION": "TOR", "SIDE": "offense", "PLAY_TYPE": "A", "PTS": 10000, "POSS": 1},
    ])
    ranked = pd.DataFrame([
        {"PLAY_TYPE": "A", "PPP_PRED": 2.5, "POSS_OFF": 100},
        {"PLAY_TYPE": "B", "PPP_PRED": 1.9, "POSS_OFF": 300},
    ])
    return team, ranked


def calculate(team, ranked, k=2):
    return compute_topk_uplift(
        team, ranked, season="2024-25", our_team="TOR", opp_team="BOS",
        k=k, w_off=0.7, provenance={"source_file": "fixture.csv"},
    )


def test_weighted_uplift_has_hand_computed_numerators_denominators(source_tables):
    payload = calculate(*source_tables)
    assert payload["team_season"]["points_numerator"] == 700
    assert payload["team_season"]["possessions_denominator"] == 400
    assert payload["team_season"]["ppp"] == 1.75
    assert payload["topk"]["modeled_points_numerator"] == 820
    assert payload["topk"]["possessions_denominator"] == 400
    assert payload["topk"]["modeled_ppp"] == pytest.approx(2.05)
    assert payload["uplift_ppp"] == pytest.approx(0.30)
    assert payload["uplift_percent"] == pytest.approx(100 * 0.30 / 1.75)
    assert [row["topk_weight"] for row in payload["rankings"]] == [0.25, 0.75]


def test_k_one_keeps_entire_team_season_baseline(source_tables):
    team, ranked = source_tables
    payload = calculate(team, ranked.head(1), k=1)
    assert payload["team_season"]["ppp"] == 1.75
    assert payload["team_season"]["play_type_rows"] == 2
    assert payload["topk"]["modeled_ppp"] == 2.5
    assert payload["uplift_ppp"] == 0.75


def test_fewer_available_types_than_k_is_explicit(source_tables):
    payload = calculate(*source_tables, k=5)
    assert payload["k"] == 5
    assert payload["k_returned"] == 2
    assert len(payload["rankings"]) == 2


def test_negative_uplift_is_retained(source_tables):
    team, ranked = source_tables
    ranked["PPP_PRED"] = 1.0
    payload = calculate(team, ranked)
    assert payload["uplift_ppp"] == -0.75


def test_zero_reference_has_null_relative_uplift_not_infinity(source_tables):
    team, ranked = source_tables
    team["PTS"] = 0
    payload = calculate(team, ranked)
    assert payload["uplift_percent"] is None
    assert "zero" in payload["relative_uplift_unavailable_reason"]
    assert payload["uplift_ppp"] == pytest.approx(2.05)


@pytest.mark.parametrize("problem", ["missing_points", "zero_possessions", "empty_rankings", "nonfinite_prediction"])
def test_unusable_sources_are_rejected_without_silent_fallback(source_tables, problem):
    team, ranked = source_tables
    if problem == "missing_points":
        team = team.drop(columns=["PTS"])
    elif problem == "zero_possessions":
        ranked["POSS_OFF"] = 0
    elif problem == "empty_rankings":
        ranked = ranked.iloc[:0]
    else:
        ranked.loc[0, "PPP_PRED"] = float("nan")
    with pytest.raises(UpliftDataUnavailable):
        calculate(team, ranked)


@pytest.fixture
def api():
    # The loaded production baseline table supplies a real end-to-end data slice.
    from application.api_coordination.app import rec, SYNERGY_CSV

    app = FastAPI()
    app.include_router(create_topk_uplift_router(rec, SYNERGY_CSV))
    with TestClient(app) as client:
        yield app, client, SYNERGY_CSV


PARAMS = {"season": "2019-20", "our": "TOR", "opp": "BOS", "k": 3}
PATHS = ["/metrics/topk-uplift", "/metrics/topk-uplift.csv", "/metrics/topk-uplift.json"]


@pytest.mark.parametrize("path", PATHS)
def test_uplift_and_downloads_require_authentication(api, path):
    _, client, _ = api
    assert client.get(path, params=PARAMS).status_code == 401


@pytest.mark.parametrize("path", PATHS)
def test_coach_cannot_access_analyst_evidence(api, path):
    app, client, _ = api
    app.dependency_overrides[require_auth] = lambda: {"role": "coach", "token": "fixture"}
    assert client.get(path, params=PARAMS).status_code == 403


def test_analyst_json_csv_match_and_include_reproducible_source(api):
    app, client, source = api
    app.dependency_overrides[require_auth] = lambda: {"role": "analyst", "token": "fixture"}
    response = client.get(PATHS[0], params=PARAMS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["k_returned"] == 3
    with source.open("rb") as handle:
        assert payload["data_provenance"]["source_sha256"] == hashlib.file_digest(handle, "sha256").hexdigest()
    assert payload["estimand"] == "historical modeled recommendation comparison"
    assert "not held-out" in payload["evaluation_scope"]
    assert payload["team_season"]["ppp"] == pytest.approx(
        payload["team_season"]["points_numerator"] / payload["team_season"]["possessions_denominator"]
    )
    assert payload["topk"]["modeled_ppp"] == pytest.approx(
        sum(row["modeled_ppp"] * row["topk_weight"] for row in payload["rankings"])
    )
    exported_json = client.get(PATHS[2], params=PARAMS)
    assert exported_json.status_code == 200
    assert exported_json.json() == payload
    assert 'attachment; filename="topk_uplift_2019-20_TOR_vs_BOS_top3.json"' == exported_json.headers["content-disposition"]
    exported_csv = client.get(PATHS[1], params=PARAMS)
    assert exported_csv.status_code == 200
    assert exported_csv.headers["content-type"].startswith("text/csv")
    assert exported_csv.headers["content-disposition"].endswith('_top3.csv"')
    rows = list(csv.DictReader(io.StringIO(exported_csv.text)))
    assert len(rows) == 3
    for row, original in zip(rows, payload["rankings"]):
        assert row["play_type"] == original["play_type"]
        assert float(row["modeled_ppp"]) == original["modeled_ppp"]
        assert float(row["uplift_ppp"]) == payload["uplift_ppp"]
        assert float(row["team_season_points_numerator"]) == payload["team_season"]["points_numerator"]
        assert row["data_provenance_source_sha256"] == payload["data_provenance"]["source_sha256"]


@pytest.mark.parametrize("override,status", [
    ({"k": 0}, 422), ({"k": 11}, 422), ({"w_off": "nan"}, 422),
    ({"opp": "TOR"}, 400), ({"our": "UNKNOWN"}, 400), ({"season": "unknown"}, 400),
])
def test_invalid_metric_filters_are_rejected(api, override, status):
    app, client, _ = api
    app.dependency_overrides[require_auth] = lambda: {"role": "analyst", "token": "fixture"}
    assert client.get(PATHS[0], params={**PARAMS, **override}).status_code == status


def test_empty_recommendations_have_no_exportable_evidence(api, monkeypatch):
    app, client, _ = api
    app.dependency_overrides[require_auth] = lambda: {"role": "analyst", "token": "fixture"}
    monkeypatch.setattr(
        "application.api_coordination.topk_uplift_endpoints.rank_playtypes_baseline",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    response = client.get(PATHS[1], params=PARAMS)
    assert response.status_code == 404
    assert "eligible recommendations" in response.json()["detail"]
