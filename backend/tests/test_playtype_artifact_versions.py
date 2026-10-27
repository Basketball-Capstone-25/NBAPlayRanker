"""Derived play-type artifacts must describe the inputs they were built with."""

import json

import pandas as pd
import pytest

from domain.statistical_analysis import ml_stat_analysis as analysis
from infrastructure.model_management.ml_models import validate_prediction_provenance


def source_frame():
    frame = pd.DataFrame({"POSS": [100], "PPP": [1.0]})
    frame.attrs.update(dataset_version="sha256:" + "a" * 64, aggregation_version="synergy_regular_season_v2", season_phase="regular_season")
    return frame


def test_predictions_reject_old_source_and_old_aggregation():
    frame = source_frame()
    predictions = pd.DataFrame({"SOURCE_DATASET_VERSION": [frame.attrs["dataset_version"]],
                                "AGGREGATION_VERSION": ["synergy_regular_season_v2"], "SEASON_PHASE": ["regular_season"]})
    validate_prediction_provenance(predictions, frame)
    for column, wrong in [("SOURCE_DATASET_VERSION", "sha256:" + "b" * 64), ("AGGREGATION_VERSION", "v1"), ("SEASON_PHASE", "playoffs")]:
        with pytest.raises(ValueError, match="rebuild"):
            validate_prediction_provenance(predictions.assign(**{column: wrong}), frame)
    with pytest.raises(ValueError, match="rebuild"):
        validate_prediction_provenance(pd.DataFrame({"PPP_ML": [1.0]}), frame)


def test_analysis_cache_signature_tracks_values_and_aggregation():
    team = source_frame()
    league = source_frame()
    signature = analysis._analysis_signature(team, league)
    changed = team.assign(PPP=[1.1])
    assert analysis._analysis_signature(changed, league) != signature
    team.attrs["aggregation_version"] = "different"
    assert analysis._analysis_signature(team, league) != signature


def test_disk_analysis_cache_rejects_wrong_source_or_legacy_schema(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setattr(analysis, "ANALYSIS_CACHE_PATH", path)
    analysis._write_disk_cache(5, 25, {"value": "current"}, "current-signature")
    assert analysis._read_disk_cache(5, 25, "current-signature") == {"value": "current"}
    assert analysis._read_disk_cache(5, 25, "different-signature") is None
    raw = json.loads(path.read_text())
    raw["schema_version"] = "ml_analysis_v2"
    path.write_text(json.dumps(raw))
    assert analysis._read_disk_cache(5, 25, "current-signature") is None
