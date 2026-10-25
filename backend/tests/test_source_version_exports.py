"""Downloads keep the versions of the rows they contain, including old snapshots."""

import csv
import io
import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader

from application.api_coordination import app as app_module
from application.api_coordination.auth_dependency import require_auth
from application.api_coordination.export_report_endpoints import create_export_report_router
from infrastructure.data_access.dataset_metadata import dataset_catalog, describe_dataset, read_versioned_csv, source_version

AUTH = "application.access_control_services.access_control_service"
OLD_VERSIONS = {"synergy": "sha256:" + "a" * 64, "ml_predictions": "sha256:" + "b" * 64}


def call(method, path, **kwargs):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH}.get_profile_role", return_value="coach"
    ):
        return TestClient(app_module.app).request(method, path, headers={"Authorization": "Bearer test"}, **kwargs)


def test_ordinary_baseline_csv_uses_the_loaded_source_version():
    response = call("GET", "/rank-plays/baseline.csv", params={"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 3})
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 3
    assert {row["DATASET_VERSION"] for row in rows} == {app_module.rec.raw_df.attrs["dataset_version"]}
    assert response.headers["X-Dataset-Version"] == rows[0]["DATASET_VERSION"]


def test_context_response_versions_come_from_the_ranker_sources(monkeypatch):
    from domain.context_ml_recommendation import ml_context_recommender as ranker
    unrelated_cache = pd.DataFrame()
    unrelated_cache.attrs["dataset_version"] = "sha256:" + "c" * 64
    monkeypatch.setattr(app_module, "ML_PRED_DF", unrelated_cache)
    response = call("GET", "/rank-plays/context-ml", params={"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 3,
                                                               "margin": -2, "period": 4, "time_remaining": 28})
    assert response.status_code == 200
    versions = {"synergy": ranker._rec.raw_df.attrs["dataset_version"],
                "ml_predictions": ranker._ML_PRED_DF.attrs["dataset_version"]}
    assert response.json()["dataset_versions"] == versions
    assert json.loads(response.headers["X-Dataset-Versions"]) == versions
    assert versions["ml_predictions"] != unrelated_cache.attrs["dataset_version"]


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_displayed_export_preserves_older_versions_in_file_and_headers(fmt):
    payload = {"report": "context_rankings", "filters": {"season": "2024-25"},
               "columns": ["Play type", "PPP"], "rows": [{"Play type": "Spotup", "PPP": 1.23}],
               "metadata": {"dataset_versions": OLD_VERSIONS}}
    response = call("POST", "/export/report", json={"format": fmt, "payload": payload})
    assert response.status_code == 200
    assert json.loads(response.headers["X-Dataset-Versions"]) == OLD_VERSIONS
    assert response.headers["X-Dataset-Version"] == OLD_VERSIONS["synergy"]
    if fmt == "json":
        assert response.json()["metadata"]["dataset_versions"] == OLD_VERSIONS
    elif fmt == "csv":
        row = next(csv.DictReader(io.StringIO(response.text)))
        assert json.loads(row["Dataset versions"]) == OLD_VERSIONS
    else:
        text = "".join(page.extract_text() for page in PdfReader(io.BytesIO(response.content)).pages)
        compact = "".join(text.split())
        assert "Datasetversions:" in compact
        assert all(f'"{key}":"{version}"' in compact for key, version in OLD_VERSIONS.items())


def test_unversioned_snapshot_does_not_claim_current_source_headers():
    payload = {"report": "baseline_rankings", "filters": {}, "columns": ["PPP"],
               "rows": [{"PPP": 1.1}], "metadata": {"dataset_versions": "Unavailable"}}
    response = call("POST", "/export/report", json={"format": "json", "payload": payload})
    assert response.status_code == 200
    assert "X-Dataset-Versions" not in response.headers
    assert response.json()["metadata"]["dataset_versions"] == "Unavailable"


def test_generated_report_uses_loaded_bytes_when_disk_source_is_different(tmp_path):
    # The router must not reopen a replacement file and label existing rows with it.
    replacement = tmp_path / "synergy.csv"
    replacement.write_text("different,data\n1,2\n")
    app = FastAPI()
    app.include_router(create_export_report_router(app_module.rec, app_module.VALID_SEASONS, app_module.VALID_TEAMS, replacement))
    app.dependency_overrides[require_auth] = lambda: {"role": "coach"}
    result = TestClient(app).get("/export/report", params={"report": "baseline_rankings", "format": "json", "season": "2024-25", "our": "TOR", "opp": "BOS"})
    assert result.status_code == 200
    expected = app_module.rec.raw_df.attrs["dataset_version"]
    assert expected != source_version(replacement)
    assert result.json()["metadata"]["dataset_versions"] == {"synergy": expected}


def test_missing_source_and_auth_errors_remain_readable_through_cors(monkeypatch):
    client = TestClient(app_module.app)
    origin = app_module._cors_origins()[0]
    unauthorized = client.get("/meta/datasets", headers={"Origin": origin})
    assert unauthorized.status_code == 401
    assert unauthorized.headers["access-control-allow-origin"] == origin
    monkeypatch.setattr(app_module, "rec", None)
    preflight = client.options("/rank-plays/baseline", headers={"Origin": origin, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"})
    assert preflight.status_code == 200
    failure = client.get("/rank-plays/baseline", headers={"Origin": origin})
    assert failure.status_code == 503
    assert failure.headers["access-control-allow-origin"] == origin
    assert "X-Dataset-Version" not in failure.headers
    assert client.get("/meta/datasets", headers={"Origin": origin}).status_code == 401


@pytest.mark.parametrize("metadata", [[], None, "unexpected", {"pulled_at": []}, {"pulled_at": "not a date"}])
def test_invalid_optional_sidecar_does_not_crash_catalog(tmp_path, metadata):
    path = tmp_path / "synergy_playtypes_2019_2025_players.csv"
    path.write_text("SEASON,POSS\n2024-25,10\n")
    if isinstance(metadata, dict):
        metadata["version"] = source_version(path)
    path.with_suffix(".csv.source.json").write_text(json.dumps(metadata))
    result = dataset_catalog(tmp_path)
    assert result[0]["status"] == "available"
    assert result[0]["pulled_at"] is None


def test_corrupt_optional_dataset_does_not_hide_other_sources(tmp_path):
    (tmp_path / "synergy_playtypes_2019_2025_players.csv").write_text("SEASON,POSS\n2024-25,10\n")
    (tmp_path / "pbp").mkdir()
    (tmp_path / "pbp/shots_clean.parquet").write_bytes(b"not a parquet file")
    result = {item["id"]: item for item in dataset_catalog(tmp_path)}
    assert result["synergy"]["status"] == "available"
    assert result["shots"]["status"] == "unavailable"


def test_parquet_metadata_reads_only_the_season_column(tmp_path, monkeypatch):
    (tmp_path / "pbp").mkdir()
    path = tmp_path / "pbp/shots_clean.parquet"
    pd.DataFrame({"SEASON_STR": ["2024-25", "2023-24"], "LARGE_VALUE": [100, 200]}).to_parquet(path)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("Full parquet read is not needed for metadata"))
    result = describe_dataset("shots", tmp_path)
    assert result["row_count"] == 2
    assert result["seasons"] == ["2023-24", "2024-25"]
    assert result["columns"] == ["SEASON_STR", "LARGE_VALUE"]
    assert result["version"] == source_version(path)


def test_catalog_refreshes_sidecar_and_preserves_loaded_version_after_file_removal(tmp_path):
    path = tmp_path / "synergy_playtypes_2019_2025_players.csv"
    path.write_text("SEASON,POSS\n2024-25,10\n")
    frame = read_versioned_csv(path)
    assert dataset_catalog(tmp_path)[0]["pulled_at"] is None
    path.with_suffix(".csv.source.json").write_text(json.dumps({"version": frame.attrs["dataset_version"], "pulled_at": "2026-10-25"}))
    assert dataset_catalog(tmp_path)[0]["pulled_at"] == "2026-10-25"
    path.unlink()
    assert describe_dataset("synergy", tmp_path, frame)["version"] == frame.attrs["dataset_version"]
