"""A failed shot snapshot save keeps the previous pair of files (SCRUM-823)."""

from pathlib import Path

import pandas as pd
import pytest

from infrastructure.data_access import pbp_snapshot
from infrastructure.data_access.dataset_metadata import source_version
from infrastructure.data_access.pbp_cache import build_meta, fingerprint_file, read_json


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "shots_clean.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(source, index=False)
    frame = pd.DataFrame({"season": ["2024-25"], "team": ["TOR"]})
    metadata = build_meta(fingerprint=fingerprint_file(source, schema_version="test"))
    paths = (tmp_path / "data" / "shots_canonical.parquet", tmp_path / "meta" / "shots_meta.json")
    for path in paths:
        path.parent.mkdir()
    return frame, metadata, paths


def fail(*args, **kwargs):
    raise OSError("injected failure")


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("failure", ["metadata_write", "readback", "changed_rows", "metadata_readback", "first_replace", "second_replace"])
def test_failed_save_preserves_both_files(snapshot, monkeypatch, existing, failure):
    frame, metadata, paths = snapshot
    if existing:
        for path in paths:
            path.write_bytes(b"previous snapshot")
    if failure == "metadata_write":
        monkeypatch.setattr(pbp_snapshot, "write_json_atomic", fail)
    elif failure == "readback":
        monkeypatch.setattr(pbp_snapshot, "read_versioned_parquet", fail)
    elif failure == "changed_rows":
        monkeypatch.setattr(pbp_snapshot, "read_versioned_parquet", lambda path: frame.assign(team="BOS"))
    elif failure == "metadata_readback":
        monkeypatch.setattr(pbp_snapshot, "read_json", lambda path: {"wrong": "snapshot"})
    else:
        replace = pbp_snapshot.os.replace
        target = paths[0 if failure == "first_replace" else 1]

        def reject_publish(source, destination):
            if Path(destination) == target:
                fail()
            replace(source, destination)

        monkeypatch.setattr(pbp_snapshot.os, "replace", reject_publish)

    with pytest.raises((OSError, ValueError)):
        pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)

    for path in paths:
        if existing:
            assert path.read_bytes() == b"previous snapshot"
        else:
            assert not path.exists()
        assert not list(path.parent.glob(".shot-reload-*"))


def test_backup_failure_does_not_publish(snapshot, monkeypatch):
    frame, metadata, paths = snapshot
    for path in paths:
        path.write_bytes(b"previous snapshot")
    monkeypatch.setattr(pbp_snapshot.shutil, "copy2", fail)
    with pytest.raises(OSError):
        pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)
    assert all(path.read_bytes() == b"previous snapshot" for path in paths)


def test_changed_source_is_rejected_before_publication(snapshot):
    frame, metadata, paths = snapshot
    Path(metadata["fingerprint"]["source_path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source changed"):
        pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)
    assert not any(path.exists() for path in paths)


def test_successful_snapshot_matches_the_read_back_data_and_metadata(snapshot):
    frame, metadata, paths = snapshot
    loaded = pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)
    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded.attrs["dataset_version"] == source_version(paths[0])
    assert loaded.attrs["source_file"] == paths[0].name
    assert read_json(paths[1]) == metadata
    for path in paths:
        assert not list(path.parent.glob(".shot-reload-*"))


def test_object_text_columns_can_return_as_string_dtype(snapshot):
    frame, metadata, paths = snapshot
    frame["team"] = frame["team"].astype(object)
    loaded = pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False, check_exact=True)


def test_failed_rollback_keeps_recovery_files_and_raises_unexpected_error(snapshot, monkeypatch):
    frame, metadata, paths = snapshot
    for path in paths:
        path.write_bytes(b"previous snapshot")
    replace = pbp_snapshot.os.replace

    def reject_publish_and_rollback(source, destination):
        if Path(destination) == paths[1] or Path(source).name == "previous":
            fail()
        replace(source, destination)

    monkeypatch.setattr(pbp_snapshot.os, "replace", reject_publish_and_rollback)
    with pytest.raises(RuntimeError, match="rollback failed"):
        pbp_snapshot.save_pbp_snapshot(frame, metadata, *paths)
    backups = list(paths[0].parent.glob(".shot-reload-*/previous"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"previous snapshot"
