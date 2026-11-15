import json

import pandas as pd
import pytest

from infrastructure.data_access.shot_forecast_source import file_sha256, load_shot_forecast_source


@pytest.fixture
def source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    game = raw / "events.parquet"
    pd.DataFrame({"game_id": [1, 1], "game_date": ["2025-11-01"] * 2,
        "season": [2026, 2026], "season_type": [2, 2]}).to_parquet(game, index=False)
    clean = tmp_path / "shots.parquet"
    pd.DataFrame({"EVENT_ID": ["10", "11"], "GAME_ID": ["1", "1"], "SEASON_STR": ["2025-26"] * 2,
        "TEAM_ABBR": ["TOR", "BOS"], "OPP_ABBR": ["BOS", "TOR"], "HOME_FLAG": [True, False],
        "POINTS": [2, 0]}).to_parquet(clean, index=False)
    metadata = {"version": "sha256:" + file_sha256(clean), "raw_source_hashes": {game.name: file_sha256(game)}}
    clean.with_suffix(".parquet.source.json").write_text(json.dumps(metadata))
    return clean, raw


def test_loads_exact_sources_and_returns_only_forecast_fields(source):
    rows, provenance = load_shot_forecast_source(*source)
    assert rows.POINTS.tolist() == [2, 0]
    assert rows.GAME_DATE.tolist() == [pd.Timestamp("2025-11-01")] * 2
    assert provenance["coverage"]["metadata_games"] == 1
    assert provenance["dataset_sha256"] == file_sha256(source[0])


@pytest.mark.parametrize("which", ["clean", "raw"])
def test_changed_file_fails_before_evaluation(source, which):
    clean, raw = source
    path = clean if which == "clean" else raw / "events.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|recorded version"):
        load_shot_forecast_source(clean, raw)


@pytest.mark.parametrize("names", [{}, {"../events.parquet": "a" * 64}, {"events.parquet": "invalid"}])
def test_missing_or_unsafe_source_pins_are_rejected(source, names):
    clean, raw = source
    sidecar = clean.with_suffix(".parquet.source.json")
    metadata = json.loads(sidecar.read_text())
    metadata["raw_source_hashes"] = names
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_shot_forecast_source(clean, raw)
