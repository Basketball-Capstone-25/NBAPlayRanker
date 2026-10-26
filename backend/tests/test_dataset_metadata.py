import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from infrastructure.data_access.dataset_metadata import describe_dataset, read_versioned_csv, source_version
from infrastructure.data_access.data_dictionary import dictionary_entries
from application.api_coordination.readiness import readiness_report


def test_version_tracks_content_and_not_the_file_name(tmp_path):
    first = tmp_path / "a.csv"
    first.write_text("SEASON,POSS\n2024-25,10\n")
    other = tmp_path / "b.csv"
    other.write_bytes(first.read_bytes())
    assert source_version(first) == source_version(other)
    loaded = read_versioned_csv(first)
    assert loaded.attrs["dataset_version"] == source_version(first)
    other.write_text("SEASON,POSS\n2024-25,11\n")
    assert source_version(first) != source_version(other)


def test_catalog_counts_actual_rows_without_faking_pull_date(tmp_path):
    path = tmp_path / "synergy_playtypes_2019_2025_players.csv"
    path.write_text("SEASON,POSS\n2024-25,10\n2023-24,20\n")
    item = describe_dataset("synergy", tmp_path)
    assert item["row_count"] == 2
    assert item["seasons"] == ["2023-24", "2024-25"]
    assert item["pulled_at"] is None
    assert "not recorded" in item["warnings"][0]
    sidecar = path.with_suffix(".csv.source.json")
    sidecar.write_text(json.dumps({"version": "old-version", "pulled_at": "2026-10-25"}))
    assert describe_dataset("synergy", tmp_path)["pulled_at"] is None
    sidecar.write_text(json.dumps({"version": source_version(path), "pulled_at": "2026-10-25"}))
    assert describe_dataset("synergy", tmp_path)["pulled_at"] == "2026-10-25"


def test_missing_dataset_is_explicit_and_unknown_columns_stay_visible(tmp_path, caplog):
    assert describe_dataset("synergy", tmp_path)["status"] == "unavailable"
    path = tmp_path / "synergy_playtypes_2019_2025_players.csv"
    path.write_text("SEASON,MYSTERY\n2024-25,10\n")
    item = describe_dataset("synergy", tmp_path, pd.read_csv(path))
    assert "no version ID" in caplog.text
    definitions = dictionary_entries([item])
    assert definitions[0]["documented"]
    assert definitions[1]["column"] == "MYSTERY"
    assert not definitions[1]["documented"]


def request(path, params=None):
    with patch("application.access_control_services.access_control_service.decode_supabase_jwt", return_value={"sub": "test"}), patch(
        "application.access_control_services.access_control_service.get_profile_role", return_value="analyst"
    ):
        return TestClient(app_module.app).get(path, params=params, headers={"Authorization": "Bearer test"})


def test_catalog_dictionary_and_export_share_loaded_source_version():
    catalog = request("/meta/datasets").json()["datasets"]
    item = next(d for d in catalog if d["id"] == "synergy")
    assert item["row_count"] == len(app_module.rec.raw_df)
    assert item["version"] == app_module.rec.raw_df.attrs["dataset_version"]
    params = {"season": "2024-25", "team": "TOR", "side": "offense"}
    result = request("/data/team-playtypes", params)
    exported = request("/data/team-playtypes.csv", params)
    assert result.json()["dataset_version"] == item["version"]
    assert result.headers["X-Dataset-Version"] == item["version"]
    assert all(r["DATASET_VERSION"] == item["version"] for r in csv.DictReader(io.StringIO(exported.text)))
    columns = request("/meta/data-dictionary").json()["columns"]
    assert {c["column"] for c in columns if c["dataset"] == "synergy"} == set(item["columns"])


def test_catalog_requires_a_session():
    assert TestClient(app_module.app).get("/meta/datasets").status_code == 401
    assert TestClient(app_module.app).get("/meta/data-dictionary").status_code == 401


def test_readiness_reports_names_but_no_secret_values():
    result = readiness_report({}, source_loaded=False, predictions_loaded=False)
    assert not result["ready"]
    assert "SUPABASE_URL" in result["missing"]
    env = {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "private-test-value"}
    result = readiness_report(env, source_loaded=True, predictions_loaded=False)
    assert result["ready"]
    assert "private-test-value" not in json.dumps(result)
    assert not readiness_report({**env, "ALLOW_INSECURE_DEV_AUTH": "true"}, source_loaded=True, predictions_loaded=True)["ready"]


def test_liveness_does_not_claim_readiness(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    client = TestClient(app_module.app)
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
