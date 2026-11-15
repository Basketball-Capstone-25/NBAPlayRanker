import json
import pandas as pd
import pytest

from data.etl import run_shot_forecast_evaluation as runner


def test_refuses_to_write_results_inside_the_repository(tmp_path):
    for destination in [runner.ROOT, runner.ROOT / "docs/new-results"]:
        with pytest.raises(ValueError, match="outside the code checkout"):
            runner.run(tmp_path / "shots.parquet", tmp_path, destination)


def test_runner_saves_predictions_and_source_fingerprints(tmp_path, monkeypatch):
    source, rows = pd.DataFrame({"source": [1]}), pd.DataFrame({"prediction": [1.25]})
    monkeypatch.setattr(runner, "load_shot_forecast_source", lambda *_: (source, {"dataset_sha256": "source-pin"}))
    monkeypatch.setattr(runner, "build_pregame_rows", lambda frame: (frame, {"warmup_shots": 0}))
    monkeypatch.setattr(runner, "evaluate_pregame_forecasts", lambda *_: ({"holdout": {"season": "2025-26"}}, rows))
    output = tmp_path / "results"
    report = runner.run(tmp_path / "shots.parquet", tmp_path, output)
    saved = json.loads((output / "report.json").read_text())
    assert saved == report
    assert pd.read_csv(output / "predictions.csv").prediction.tolist() == [1.25]
    assert saved["provenance"]["predictions_sha256"] == runner.file_sha256(output / "predictions.csv")
    assert "backend/domain/shot_analysis/shot_forecast_features.py" in saved["provenance"]["source_sha256"]
    assert saved["provenance"]["dataset_sha256"] == "source-pin"


def test_failed_evaluation_does_not_create_result_files(tmp_path, monkeypatch):
    def failed(*_):
        raise ValueError("Source data changed")
    monkeypatch.setattr(runner, "load_shot_forecast_source", failed)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Source data changed"):
        runner.run(tmp_path / "shots.parquet", tmp_path, output)
    assert not output.exists()
