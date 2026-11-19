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


@pytest.mark.parametrize("rows,reason", [
    ([{**SHOT, "OPP_ABBR": "TOR"}], "1 self-opponent rows"),
    ([{**SHOT, "TEAM_ABBR": "NY", "OPP_ABBR": "NYK"}], "1 self-opponent rows"),
    ([{**SHOT, "TEAM_ABBR": "BAD"}], "1 unsupported team rows"),
    ([{**SHOT, "OPP_ABBR": None}], "1 unsupported team rows"),
    ([SHOT] * 100, "1 incomplete game groups"),
])
def test_invalid_attribution_keeps_stored_and_loaded_shots(setup, rows, reason):
    service, before, calls, tmp_path = setup
    service._canonical_path.parent.mkdir()
    before.to_parquet(service._canonical_path, index=False)
    service._meta_path.write_text('{"snapshot": "previous"}')
    stored = service._canonical_path.read_bytes()
    metadata = service._meta_path.read_bytes()
    pd.DataFrame(rows).to_parquet(service._clean_path, index=False)

    with pytest.raises(DataReloadError, match=reason):
        service.reload()

    assert service._canonical_path.read_bytes() == stored
    assert service._meta_path.read_bytes() == metadata
    assert pbp_loader.loaded_pbp_canonical() is before
    assert pbp_loader._CANONICAL_ID == before.attrs["dataset_version"]
    assert calls == []


def test_two_team_game_passes_attribution_check(setup):
    service, _, calls, _ = setup
    other_team = {**SHOT, "TEAM_ABBR": "BOS", "OPP_ABBR": "TOR"}
    pd.DataFrame([SHOT, other_team] * 60).to_parquet(service._clean_path, index=False)

    assert service.reload()["rows"] == 120
    assert calls == [True]


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("failure", ["metadata_write", "readback", "second_publish"])
def test_snapshot_failure_keeps_disk_and_memory_together(setup, monkeypatch, existing, failure):
    from infrastructure.data_access import pbp_snapshot

    service, before, calls, _ = setup
    paths = (service._canonical_path, service._meta_path)
    if existing:
        paths[0].parent.mkdir()
        for path in paths:
            path.write_bytes(b"previous snapshot")
    pd.DataFrame([SHOT]).to_parquet(service._clean_path, index=False)

    def fail(*args, **kwargs):
        raise OSError("injected snapshot failure")

    if failure == "metadata_write":
        monkeypatch.setattr(pbp_snapshot, "write_json_atomic", fail)
    elif failure == "readback":
        monkeypatch.setattr(pbp_snapshot, "read_versioned_parquet", fail)
    else:
        replace = pbp_snapshot.os.replace

        def reject_metadata(source, target):
            if target == service._meta_path:
                fail()
            replace(source, target)

        monkeypatch.setattr(pbp_snapshot.os, "replace", reject_metadata)

    with pytest.raises(DataReloadError, match="could not be saved"):
        service.reload()
    for path in paths:
        if existing:
            assert path.read_bytes() == b"previous snapshot"
        else:
            assert not path.exists()
    assert pbp_loader.loaded_pbp_canonical() is before
    assert pbp_loader._CANONICAL_ID == before.attrs["dataset_version"]
    assert calls == []


def test_source_changed_after_read_is_not_used_for_metadata(setup, monkeypatch):
    from application.data_reload_services import pbp_reload

    service, before, calls, _ = setup
    pd.DataFrame([SHOT]).to_parquet(service._clean_path, index=False)
    convert = pbp_reload.build_canonical_from_clean

    def replace_source(clean):
        pd.DataFrame([{**SHOT, "GAME_ID": "new-game"}]).to_parquet(service._clean_path, index=False)
        return convert(clean)

    monkeypatch.setattr(pbp_reload, "build_canonical_from_clean", replace_source)
    with pytest.raises(DataReloadError, match="could not be saved"):
        service.reload()
    assert not service._canonical_path.exists() and not service._meta_path.exists()
    assert pbp_loader.loaded_pbp_canonical() is before and calls == []


def test_source_frame_is_released_before_snapshot_readback(setup, monkeypatch):
    import weakref
    from application.data_reload_services import pbp_reload

    service, _, _, _ = setup
    pd.DataFrame([SHOT]).to_parquet(service._clean_path, index=False)
    read = pbp_reload.read_versioned_parquet
    save = pbp_reload.save_pbp_snapshot
    references = []

    def track_source(path):
        frame = read(path)
        references.append(weakref.ref(frame))
        return frame

    def check_source_released(*args):
        assert references[0]() is None
        return save(*args)

    monkeypatch.setattr(pbp_reload, "read_versioned_parquet", track_source)
    monkeypatch.setattr(pbp_reload, "save_pbp_snapshot", check_source_released)
    assert service.reload()["rows"] == 1


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
