"""Runtime shot data must work from the distributed clean dataset alone."""

import pandas as pd
import pytest

from infrastructure.data_access import pbp_clean


@pytest.fixture
def clean_dataset(tmp_path, monkeypatch):
    for name, filename in {
        "CLEAN_PARQUET": "shots_clean.parquet",
        "SOURCE_PARQUET": "raw_missing.parquet",
        "ALT_SOURCE_PARQUET": "alternate_raw_missing.parquet",
        "CANONICAL_PARQUET": "cache/shots_canonical.parquet",
        "CANONICAL_META_JSON": "cache/shots_canonical_meta.json",
    }.items():
        monkeypatch.setattr(pbp_clean, name, tmp_path / filename)
    monkeypatch.setattr(pbp_clean, "CACHE_DIR", tmp_path / "cache")
    frame = pd.DataFrame([{
        "SEASON_STR": "2024-25", "TEAM_ABBR": "TOR", "OPP_ABBR": "BOS",
        "GAME_ID": "game-1", "X": 1.0, "Y": 4.0, "POINTS": 2,
        "MADE": 1, "SHOT_VALUE": 2, "SHOT_TYPE": "Jump Shot",
        "ZONE": "paint", "HOME_FLAG": True,
    }])
    frame.to_parquet(pbp_clean.CLEAN_PARQUET, index=False)
    return frame


def test_canonical_build_and_cache_work_without_raw_download(clean_dataset):
    path = pbp_clean.ensure_canonical_parquet()
    result = pd.read_parquet(path)
    assert result.loc[0, "team"] == "TOR"
    assert result.loc[0, "points"] == 2
    original_mtime = path.stat().st_mtime_ns

    assert pbp_clean.ensure_canonical_parquet() == path
    assert path.stat().st_mtime_ns == original_mtime
    assert not pbp_clean.SOURCE_PARQUET.exists()


def test_canonical_refreshes_when_clean_input_changes(clean_dataset):
    path = pbp_clean.ensure_canonical_parquet()
    updated = pd.concat([clean_dataset, clean_dataset.assign(GAME_ID="game-2")])
    updated.to_parquet(pbp_clean.CLEAN_PARQUET, index=False)

    assert pbp_clean.ensure_canonical_parquet() == path
    assert pd.read_parquet(path)["game_id"].tolist() == ["game-1", "game-2"]
