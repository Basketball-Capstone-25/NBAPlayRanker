import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module, shot_data_guard
from application.api_coordination.auth_dependency import require_auth
from application.recommendation_services import export_shotplan_pdf


CANONICAL_TEAMS = {"GSW", "SAS", "NOP", "NYK", "UTA", "WAS", "PHX", "TOR"}


@pytest.fixture
def shot_client(monkeypatch):
    calls = []

    class CanonicalShotRecommender:
        agg_df = pd.DataFrame()
        agg_df.attrs.update(source_file="loaded-shot-aggregates.parquet", dataset_version="sha256:" + "a" * 64)
        league_df = pd.DataFrame()
        league_df.attrs.update(source_file="loaded-shot-league.parquet", dataset_version="sha256:" + "b" * 64)

        def rank(self, *, season, our_team, opp_team, k, w_off):
            if our_team not in CANONICAL_TEAMS or opp_team not in CANONICAL_TEAMS:
                raise ValueError("Unknown shot team")
            calls.append((our_team, opp_team))
            return {"top_shot_types": [], "top_zones": []}

    def pdf(**kwargs):
        CanonicalShotRecommender().rank(
            season=kwargs["season"], our_team=kwargs["our"], opp_team=kwargs["opp"],
            k=kwargs["k"], w_off=kwargs["w_off"],
        )
        return io.BytesIO(b"%PDF-1.4\nfixture"), "shot-plan.pdf"

    monkeypatch.setattr(shot_data_guard, "shot_data_quality", lambda: {"valid_attribution": True})
    monkeypatch.setattr(app_module, "_get_shot_rec", CanonicalShotRecommender)
    monkeypatch.setattr(export_shotplan_pdf, "build_shotplan_pdf", pdf)
    app_module.app.dependency_overrides[require_auth] = lambda: {"role": "analyst"}
    try:
        yield TestClient(app_module.app), calls
    finally:
        app_module.app.dependency_overrides.pop(require_auth, None)


@pytest.mark.parametrize("path", ["/shotplan/rank", "/export/shotplan.pdf"])
@pytest.mark.parametrize("team", ["GSW", "SAS", "NOP", "NYK", "UTA", "WAS", "PHX"])
def test_root_shot_routes_preserve_canonical_team_codes(shot_client, path, team):
    client, calls = shot_client
    response = client.get(path, params={"season": "2024-25", "our": team, "opp": "TOR"})
    assert response.status_code == 200
    assert calls == [(team, "TOR")]
    if path.endswith(".pdf"):
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
    else:
        assert response.json()["our_team"] == team
        assert response.json()["metadata"] == {
            "data_source": "loaded-shot-aggregates.parquet",
            "league_source": "loaded-shot-league.parquet",
            "dataset_versions": {"shot_aggregates": "sha256:" + "a" * 64, "shot_league": "sha256:" + "b" * 64},
        }


@pytest.mark.parametrize("path", ["/shotplan/rank", "/export/shotplan.pdf"])
@pytest.mark.parametrize("alias,canonical", [
    (" gs ", "GSW"), ("sa", "SAS"), ("NO", "NOP"), ("NY", "NYK"),
    ("UTAH", "UTA"), ("WSH", "WAS"), ("PHO", "PHX"),
])
def test_older_shot_aliases_resolve_toward_canonical_codes(shot_client, path, alias, canonical):
    client, calls = shot_client
    response = client.get(path, params={"season": "2024-25", "our": "TOR", "opp": alias})
    assert response.status_code == 200
    assert calls == [("TOR", canonical)]
    if not path.endswith(".pdf"):
        assert response.json()["opp_team"] == canonical


def test_shot_alias_change_does_not_rewrite_synergy_team_codes(shot_client):
    client, calls = shot_client
    app_module.app.dependency_overrides[require_auth] = lambda: {"role": "coach"}
    response = client.get("/rank-plays/baseline", params={
        "season": "2024-25", "our": "GSW", "opp": "TOR", "k": 5,
    })
    assert response.status_code == 200
    assert response.json()["our_team"] == "GSW"
    assert len(response.json()["rankings"]) == 5
    assert calls == []
