"""Save the checked shot table and its metadata together (SCRUM-823)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .dataset_metadata import read_versioned_parquet, source_version
from .pbp_cache import read_json, write_json_atomic, write_parquet_atomic


def save_pbp_snapshot(
    frame: pd.DataFrame, metadata: dict, canonical_path: Path, meta_path: Path,
) -> pd.DataFrame:
    """Check staged files first; roll back a failed replacement of either file.

    Backups cover caught filesystem errors, not a process crash between renames.
    The caller must serialize reloads and swap memory only after this returns.
    """
    targets = (Path(canonical_path), Path(meta_path))
    folders: list[Path] = []
    keep_backups = False
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            folders.append(Path(tempfile.mkdtemp(prefix=".shot-reload-", dir=target.parent)))
        staged = [folder / target.name for folder, target in zip(folders, targets)]
        write_parquet_atomic(frame, staged[0])
        write_json_atomic(staged[1], metadata)
        loaded = read_versioned_parquet(staged[0])
        try:
            # Parquet can return an object text column as pandas' string dtype.
            pd.testing.assert_frame_equal(loaded, frame, check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("The staged shot table did not match the checked rows.") from error
        if read_json(staged[1]) != metadata:
            raise ValueError("The staged shot files did not match the checked snapshot.")
        fingerprint = metadata["fingerprint"]
        if source_version(Path(fingerprint["source_path"])) != fingerprint["source_sha256"]:
            raise ValueError("The shot source changed while preparing the reload.")

        backups: list[Path | None] = []
        for target, folder in zip(targets, folders):
            backup = folder / "previous" if target.exists() else None
            if backup is not None:
                shutil.copy2(target, backup)
            backups.append(backup)

        replaced: list[int] = []
        try:
            for index, (candidate, target) in enumerate(zip(staged, targets)):
                os.replace(candidate, target)
                replaced.append(index)
        except OSError:
            for index in reversed(replaced):
                try:
                    if backups[index] is None:
                        targets[index].unlink(missing_ok=True)
                    else:
                        os.replace(backups[index], targets[index])
                except OSError as error:
                    keep_backups = True
                    # This is not a safe rejection: keep recovery files and let
                    # the caller return an unexpected error instead of HTTP 422.
                    raise RuntimeError(f"Shot rollback failed; backups kept in {folders}.") from error
            raise
        return loaded
    finally:
        if not keep_backups:
            for folder in folders:
                shutil.rmtree(folder, ignore_errors=True)
