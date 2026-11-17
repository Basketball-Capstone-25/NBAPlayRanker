"""Analyst play-by-play reload keeps the old shots unless the new file passes (SCRUM-491)."""

from __future__ import annotations

import pandas as pd
import pytest

from application.data_reload_services.pbp_reload import PbpReloadService
from application.data_reload_services.synergy_reload import DataReloadError, ReloadInProgress
from infrastructure.data_access import pbp_loader

SHOT = {
    "SEASON_STR": "2024-25", "TEAM_ABBR": "TOR", "OPP_ABBR": "BOS", "GAME_ID": "game-1",
    "X": 1.0, "Y": 4.0, "POINTS": 2, "MADE": 1, "SHOT_VALUE": 2, "SHOT_TYPE": "Jump Shot",
    "ZONE": "paint", "HOME_FLAG": True, "PERIOD": 4, "CLOCK_SEC": 30, "MARGIN": -2,
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    before = pd.DataFrame({"season": ["2023-24"], "team": ["NYK"]})
    before.attrs["dataset_version"] = "sha256:" + "1" * 64
    monkeypatch.setattr(pbp_loader, "_CANONICAL_DF", before)
    monkeypatch.setattr(pbp_loader, "_CANONICAL_ID", before.attrs["dataset_version"])
    calls = []
    service = PbpReloadService(
        clean_path=tmp_path / "shots_clean.parquet",
        canonical_path=tmp_path / "cache" / "shots_canonical.parquet",
        meta_path=tmp_path / "cache" / "shots_canonical_meta.json",
        on_reloaded=lambda: calls.append(True),
    )
    return service, before, calls, tmp_path


def test_successful_reload_swaps_in_the_new_shots(setup):
    service, before, calls, tmp_path = setup
    pd.DataFrame([SHOT, {**SHOT, "GAME_ID": "game-2", "POINTS": 0, "MADE": 0}]).to_parquet(tmp_path / "shots_clean.parquet", index=False)

    result = service.reload()

    loaded = pbp_loader.get_pbp_canonical_df()
    assert loaded is not before and loaded["game_id"].tolist() == ["game-1", "game-2"]
    assert result["status"] == "reloaded" and result["rows"] == 2 and result["seasons"] == ["2024-25"]
    assert result["previous_version"] == before.attrs["dataset_version"]
    assert result["dataset_version"] == loaded.attrs["dataset_version"] and result["changed"] is True
    assert (tmp_path / "cache" / "shots_canonical_meta.json").exists()
    assert calls == [True]


@pytest.mark.parametrize("write,message", [
    (None, "shots_clean.parquet was not found"),
    (lambda path: path.write_text("not parquet"), "shots_clean.parquet could not be read"),
    (lambda path: pd.DataFrame([{k: v for k, v in SHOT.items() if k != "GAME_ID"}]).to_parquet(path, index=False),
     "missing columns: GAME_ID"),
    (lambda path: pd.DataFrame([{**SHOT, "X": None}]).to_parquet(path, index=False), "has no usable shots"),
    (lambda path: pd.DataFrame([{**SHOT, "SEASON_STR": "season one"}]).to_parquet(path, index=False), "unrecognised seasons"),
])
def test_failed_reload_keeps_previous_shots(setup, write, message):
    service, before, calls, tmp_path = setup
    if write is not None:
        write(tmp_path / "shots_clean.parquet")

    with pytest.raises(DataReloadError, match=message):
        service.reload()

    assert pbp_loader.get_pbp_canonical_df() is before
    assert not (tmp_path / "cache" / "shots_canonical.parquet").exists()
    assert calls == []


def test_second_reload_waits_for_the_first(setup):
    service = setup[0]
    service._lock.acquire()
    try:
        with pytest.raises(ReloadInProgress):
            service.reload()
    finally:
        service._lock.release()


def test_shot_views_keep_loaded_rows_until_the_reload_clears_them(monkeypatch, tmp_path):
    """SCRUM-491: heatmap/PDF and shot model rows do not follow the file on their own."""
    from application.api_coordination import app as app_module
    from domain.shot_analysis import shot_ml_models
    from infrastructure.data_access import shot_plot_data

    plot = pd.DataFrame({"SEASON_STR": ["2024-25"]})
    monkeypatch.setattr(shot_plot_data, "_FRAME", plot)
    monkeypatch.setattr(shot_plot_data, "CLEAN_PARQUET", tmp_path / "missing.parquet")
    cached = shot_ml_models._CleanCache(cache_id="loaded", df=pd.DataFrame({"SEASON_STR": ["2024-25"]}))
    monkeypatch.setattr(shot_ml_models, "_CLEAN_CACHE", cached)
    monkeypatch.setattr(shot_ml_models, "CLEAN_PARQUET", tmp_path / "missing.parquet")

    assert shot_plot_data.get_shot_plot_df() is plot
    assert shot_ml_models.get_shots_clean_df() is cached.df

    app_module._after_pbp_reload()
    assert shot_plot_data._FRAME is None and shot_ml_models._CLEAN_CACHE is None


def test_shot_check_describes_the_loaded_rows_until_a_reload(monkeypatch):
    """SCRUM-491: a broken file on disk does not pause shot results on its own."""
    from application.api_coordination import app as app_module
    from domain.shot_analysis import shot_quality

    loaded = {"valid_attribution": True}
    monkeypatch.setattr(shot_quality, "_LOADED", loaded)
    assert shot_quality.shot_data_quality() is loaded
    app_module._after_pbp_reload()
    assert shot_quality._LOADED is None
