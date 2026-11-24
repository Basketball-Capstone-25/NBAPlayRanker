import json
import os

import pytest

from data.etl import record_synergy_source as recorder
from infrastructure.data_access.dataset_metadata import describe_dataset


def _chunk(folder, name, when):
    path = folder / name
    path.write_text("SEASON\n2024-25\n")
    os.utime(path, (when, when))


def test_pull_date_is_the_newest_chunk_download(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    _chunk(chunks, "P_2024-25_RS_Off_Cut.csv", 1_765_130_000)
    _chunk(chunks, "P_2025-26_RS_Off_Cut.csv", 1_795_000_000)
    csv_path = tmp_path / "synergy.csv"
    csv_path.write_text("SEASON,POSS\n2024-25,10\n")
    record = recorder.build_record(csv_path, chunks)
    assert record["pulled_at"] == "2026-11-18T11:06:40+00:00"
    assert record["season_downloaded_at"]["2024-25"] < record["pulled_at"]
    assert record["chunk_count"] == 2


def test_no_chunks_means_no_pull_date(tmp_path):
    (tmp_path / "chunks").mkdir()
    csv_path = tmp_path / "synergy.csv"
    csv_path.write_text("SEASON\n2024-25\n")
    with pytest.raises(ValueError):
        recorder.build_record(csv_path, tmp_path / "chunks")


def test_explorer_shows_the_recorded_date_only_for_the_same_bytes(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    _chunk(chunks, "P_2024-25_RS_Off_Cut.csv", 1_795_000_000)
    csv_path = tmp_path / "synergy_playtypes_2019_2025_players.csv"
    csv_path.write_text("SEASON,POSS\n2024-25,10\n")
    sidecar = csv_path.with_suffix(".csv.source.json")
    sidecar.write_text(json.dumps(recorder.build_record(csv_path, chunks)))
    assert describe_dataset("synergy", tmp_path)["pulled_at"] == "2026-11-18T11:06:40+00:00"
    csv_path.write_text("SEASON,POSS\n2024-25,11\n")
    assert describe_dataset("synergy", tmp_path)["pulled_at"] is None
