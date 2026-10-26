import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.api_coordination.app import app
from application.api_coordination.auth_dependency import require_auth
from application.api_coordination import pbp_endpoints, pbp_phase2_endpoints, shot_data_guard
from domain.shot_analysis import shot_artifacts, shot_ml_models
from domain.shot_analysis.shot_etl import _parse_margin, _pick_col
from domain.shot_analysis.shot_quality import inspect_shot_rows


def test_exact_column_selection_never_uses_score_or_home_team_for_event_identity():
    df = pd.DataFrame(columns=["home_score", "home_team_abbreviation", "away_team_abbreviation"])
    assert _pick_col(df, ["team"]) is None
    assert _pick_col(df, ["home", "is_home"]) is None
    assert np.isnan(_parse_margin(""))
    assert _parse_margin("TIE") == 0


def complete_rows():
    spec = shot_ml_models.get_feature_spec()
    df = pd.DataFrame({name: [1.0, 1.0] for name in spec["numeric_features"]})
    for name in spec["categorical_features"]:
        df[name] = ["fixture", "fixture"]
    return df.assign(SEASON_STR="2024-25", TEAM_ABBR=["TOR", "BOS"], OPP_ABBR=["BOS", "TOR"], GAME_ID="1", POINTS=[0, 2], MADE=[0, 1])


def test_schema_reports_extra_metadata_and_missing_required_columns():
    df = complete_rows().drop(columns="TEAM_ABBR").assign(UNMAPPED_FIXTURE="metadata")
    report = shot_ml_models.shot_schema_report(df)
    assert report["missing_required"] == ["TEAM_ABBR"]
    assert report["ignored"] == ["UNMAPPED_FIXTURE"]
    assert "MADE" in report["optional"]
    assert "POINTS" in report["mapped"]
    assert shot_ml_models.shot_schema_report(complete_rows().assign(NOTE="metadata"))["missing_required"] == []


def test_missing_margin_is_not_imputed_as_tied_score(monkeypatch):
    df = complete_rows().assign(MARGIN=np.nan)
    monkeypatch.setattr(shot_ml_models, "get_shots_clean_df", lambda: df)
    with pytest.raises(ValueError, match="MARGIN"):
        shot_ml_models.load_shots_for_ml()
    assert df.MARGIN.isna().all()


def test_quality_checks_unknown_teams_and_substantial_one_team_games():
    assert inspect_shot_rows(complete_rows())["valid_attribution"]
    wrong = complete_rows().assign(OPP_ABBR=["TOR", "BOS"])
    assert inspect_shot_rows(wrong)["invalid_participant_rows"] == 2
    one_team = pd.concat([complete_rows().iloc[[0]]] * 100)
    assert inspect_shot_rows(one_team)["incomplete_game_groups"] == 1


def test_saved_results_require_current_data_folds_and_version(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source-one")
    artifact = {"artifact_version": shot_artifacts.ARTIFACT_VERSION, "source_sha256": shot_artifacts.source_hash(source), "n_splits": 5, "metrics": [{"model": "fixture", "RMSE_mean": .2}]}
    path = tmp_path / "pbp_phase2_shot_model_cv.json"
    path.write_text(json.dumps(artifact))
    before = path.read_bytes()
    assert shot_artifacts.read_shot_artifact("metrics", 5, cache_dir=tmp_path, source=source)["cached"] is True
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="do not match"):
        shot_artifacts.read_shot_artifact("metrics", 4, cache_dir=tmp_path, source=source)
    source.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="do not match"):
        shot_artifacts.read_shot_artifact("metrics", 5, cache_dir=tmp_path, source=source)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(shot_data_guard, "shot_data_quality", lambda: {"valid_attribution": True})
    yield TestClient(app)
    app.dependency_overrides.pop(require_auth, None)


def signed_in(role):
    app.dependency_overrides[require_auth] = lambda: {"role": role, "token": "fixture"}


@pytest.mark.parametrize("role", ["analyst", "coach"])
@pytest.mark.parametrize("path", ["/pbp/shotplan/rank", "/pbp/viz/shot-heatmap"])
def test_shot_tools_allow_analyst_and_coach_without_other_coach_routes(api, monkeypatch, role, path):
    signed_in(role)
    monkeypatch.setattr(pbp_endpoints, "get_shotplan_json", lambda **kw: {"top_shot_types": []})
    monkeypatch.setattr(pbp_endpoints, "render_pbp_heatmap_base64", lambda **kw: {"image_base64": "fixture"})
    assert api.get(path, params={"season": "2024-25", "our": "TOR", "opp": "BOS"}).status_code == 200
    if role == "analyst":
        assert api.get("/rank-plays/baseline", params={"season": "2024-25", "our": "TOR", "opp": "BOS"}).status_code == 403


@pytest.mark.parametrize("path", ["/pbp/shotplan/rank", "/pbp/viz/shot-heatmap", "/metrics/shot-models", "/pbp/metrics/shot-models"])
def test_shot_apis_deny_signed_out_and_wrong_role(api, path):
    params = {"season": "2024-25", "our": "TOR", "opp": "BOS"}
    assert api.get(path, params=params).status_code == 401
    signed_in("pending")
    assert api.get(path, params=params).status_code == 403


@pytest.mark.parametrize("path", ["/metrics/shot-models", "/pbp/metrics/shot-models"])
def test_metrics_read_saved_results_without_evaluation(api, monkeypatch, path):
    signed_in("analyst")
    def no_training(*args, **kwargs):
        pytest.fail("A page visit must not fit a model")
    monkeypatch.setattr(shot_ml_models, "run_shot_model_cv", no_training)
    monkeypatch.setattr(pbp_phase2_endpoints, "read_shot_artifact", lambda kind, n_splits: {"cached": True, "n_splits": n_splits, "metrics": []})
    result = api.get(path).json()
    assert result["cached"] and result["n_splits"] == 5


def test_missing_cache_and_refresh_fail_without_training(api, monkeypatch):
    signed_in("analyst")
    def missing(*args):
        raise ValueError("Saved shot results are not available")
    monkeypatch.setattr(pbp_phase2_endpoints, "read_shot_artifact", missing)
    assert api.get("/metrics/shot-models").status_code == 503
    assert api.get("/pbp/metrics/shot-models?refresh=true").status_code == 400


def test_invalid_dataset_stops_recommendations_and_models(api, monkeypatch):
    signed_in("analyst")
    monkeypatch.setattr(shot_data_guard, "shot_data_quality", lambda: {"valid_attribution": False})
    for path in ["/metrics/shot-models", "/pbp/shotplan/rank", "/pbp/viz/shot-heatmap"]:
        result = api.get(path, params={"season": "2024-25", "our": "TOR", "opp": "BOS"})
        assert result.status_code == 409
        assert "team labels" in result.json()["detail"]


def test_hoopr_participant_ids_margin_and_decimal_clock_are_preserved():
    from domain.shot_analysis.shot_etl import build_shots_clean
    raw = pd.DataFrame({
        "shooting_play": [True, True], "season": [2025, 2025], "game_id": ["g", "g"], "id": ["home", "away"],
        "team_id": [28, 2], "home_team_id": [28, 28], "away_team_id": [2, 2], "home_team_abbrev": ["TOR", "TOR"], "away_team_abbrev": ["BOS", "BOS"],
        "home_score": [52, 52], "away_score": [50, 53], "score_value": [2, 3], "scoring_play": [True, True],
        "points_attempted": [2, 3], "period_number": [2, 2], "clock_display_value": ["0:45.2", "0:31.3"],
        "coordinate_x": [38.75, -18.75], "coordinate_y": [0., 0.], "coordinate_x_raw": [25., 25.], "coordinate_y_raw": [3., 23.],
        "type_text": ["Layup", "Jump Shot"], "text": ["Home player makes layup", "Away player makes three point jumper"],
    })
    clean = build_shots_clean(raw).set_index("EVENT_ID")
    assert clean.loc["home", "TEAM_ABBR"] == "TOR"
    assert clean.loc["away", "TEAM_ABBR"] == "BOS"
    assert clean.loc["home", "MARGIN"] == 0
    assert clean.loc["away", "MARGIN"] == -2
    assert clean.loc["home", "CLOCK_SEC"] == 45.2
    assert clean.loc["home", "DIST"] == 3
    assert clean.loc["away", "POINTS"] == 3
    changed = raw.assign(home_score=[99, 99])
    assert build_shots_clean(changed).set_index("EVENT_ID").TEAM_ABBR.to_dict() == clean.TEAM_ABBR.to_dict()


def test_ambiguous_missed_attempt_value_remains_unknown():
    from domain.shot_analysis.shot_etl import build_shots_clean
    raw = pd.DataFrame({"shooting_play": [True], "season": [2025], "game_id": ["g"], "team_id": [1610612761],
        "home_team_abbreviation": ["TOR"], "away_team_abbreviation": ["BOS"], "x": [25.], "y": [0.],
        "text": ["Player misses jump shot"], "scoring_play": [False], "score_value": [0]})
    clean = build_shots_clean(raw)
    assert clean.SHOT_VALUE.isna().all()
    assert clean.POINTS.iloc[0] == 0
    assert "SHOT_VALUE" not in shot_ml_models.get_feature_spec()["numeric_features"]


def test_league_aggregates_keep_shot_type_and_zone_levels():
    from domain.shot_analysis.shot_aggregates import build_league_baselines, build_shot_aggregates
    rows = complete_rows().assign(SEASON_STR="2024-25", SHOT_TYPE="Jump Shot", ZONE="mid", MADE=[0, 1])
    league = build_league_baselines(build_shot_aggregates(rows))
    assert set(league.LEVEL) == {"shot_type", "zone", "shot_type_zone"}
    assert league.EPA_LEAGUE.eq(1).all()


def test_cross_validation_never_uses_the_reserved_newest_season(monkeypatch):
    old = pd.concat([complete_rows().assign(GAME_ID=str(index)) for index in range(4)], ignore_index=True)
    latest = complete_rows().assign(SEASON_STR="2025-26", DIST=999, GAME_ID="holdout")
    monkeypatch.setattr(shot_ml_models, "load_shots_for_ml", lambda **kwargs: pd.concat([old, latest], ignore_index=True))
    calls = []
    def inspect_fit(name, pipeline, X, y, train, test):
        assert not X.DIST.eq(999).any()
        assert not any(column.startswith("SEASON_STR") for column in X.columns)
        calls.append((len(train), len(test)))
        return {"model": name, "RMSE": 1., "MAE": 1., "R2": 0.}
    monkeypatch.setattr(shot_ml_models, "_fit_and_eval", inspect_fit)
    _, folds = shot_ml_models.run_shot_model_cv(n_splits=2)
    assert len(folds) == 6 and len(calls) == 6


def test_saved_analysis_response_keeps_provenance(api, monkeypatch):
    signed_in("analyst")
    saved = {"cached":True,"source_sha256":"a"*64,"n_splits":5,"artifact_version":shot_artifacts.ARTIFACT_VERSION,"payload":{"rows_used":25,"feature_spec":{"numeric_features":["MARGIN"]}}}
    before = json.dumps(saved, sort_keys=True)
    monkeypatch.setattr(pbp_phase2_endpoints,"read_shot_artifact",lambda kind,n_splits:saved)
    result = api.get("/analysis/shot-ml").json()
    assert result["rows_used"] == 25 and result["cached"] is True
    assert result["dataset_version"] == "sha256:" + "a"*64
    assert result["n_splits"] == 5 and result["artifact_version"] == shot_artifacts.ARTIFACT_VERSION
    assert json.dumps(saved, sort_keys=True) == before


@pytest.mark.parametrize("document", [[], None, "invalid"])
def test_non_object_saved_artifact_returns_clear_error(tmp_path, document):
    (tmp_path / "pbp_phase2_shot_model_cv.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="object"):
        shot_artifacts.read_shot_artifact("metrics",5,cache_dir=tmp_path)
