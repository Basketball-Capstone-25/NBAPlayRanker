"""Read the pinned shot file and original game metadata without changing either."""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from domain.shot_analysis.shot_forecast_source import GAME_COLUMNS, SHOT_COLUMNS, prepare_shot_forecast_rows


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shot_forecast_source(clean_path: Path, raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load the clean shot file only if it matches its recorded source version and raw files."""
    metadata = json.loads(clean_path.with_suffix(clean_path.suffix + ".source.json").read_text())
    expected = metadata.get("version", "")
    if not isinstance(expected, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        raise ValueError("The clean shot source must have a pinned SHA-256 version.")
    actual = file_sha256(clean_path)
    if expected != "sha256:" + actual:
        raise ValueError("Clean shot data does not match its recorded version.")
    sources = metadata.get("raw_source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Original shot sources must have pinned hashes.")
    games = []
    for name, expected_hash in sorted(sources.items()):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".parquet"):
            raise ValueError("Raw source names must be Parquet filenames within the supplied directory.")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("Each original source must have a valid SHA-256 hash.")
        path = raw_dir / name
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Original shot source changed: {name}")
        games.append(pd.read_parquet(path, columns=GAME_COLUMNS).drop_duplicates())
    shots = pd.read_parquet(clean_path, columns=SHOT_COLUMNS)
    rows, coverage = prepare_shot_forecast_rows(shots, pd.concat(games, ignore_index=True))
    return rows, {
        "coverage": coverage,
        "dataset": clean_path.name, "dataset_sha256": actual,
        "raw_source_hashes": sources,
        "source": "ESPN NBA play-by-play via hoopR / sportsdataverse-data",
    }
