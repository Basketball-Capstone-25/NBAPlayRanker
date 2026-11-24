"""Record where the play-type CSV came from and when its data was pulled.

Writes <csv>.source.json next to the CSV. The pull date is the newest chunk
download time, so it is only as accurate as the chunk files on this machine.
Run it right after build_synergy_dataset.R writes the CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
DEFAULT_CSV = BACKEND / "data/synergy_playtypes_2019_2025_players.csv"
DEFAULT_CHUNKS = BACKEND / "data/synergy_chunks"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def build_record(csv_path: Path, chunk_dir: Path) -> dict:
    """Source record for the CSV: content hash, pull date and download date per season."""
    chunks = sorted(chunk_dir.glob("*.csv"))
    if not chunks:
        raise ValueError(f"No chunk files in {chunk_dir}; the pull date cannot be recorded.")
    digest = hashlib.sha256()
    with csv_path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    version = "sha256:" + digest.hexdigest()
    seasons: dict[str, float] = {}
    for chunk in chunks:
        # Chunk names look like P_2019-20_RS_Off_Cut.csv.
        parts = chunk.stem.split("_")
        season = parts[1] if len(parts) > 1 else "unknown"
        seasons[season] = max(seasons.get(season, 0.0), chunk.stat().st_mtime)
    return {
        "version": version,
        "pulled_at": _iso(max(seasons.values())),
        "source": "NBA Stats / Synergy play types via hoopR",
        "generator": "backend/data/etl/build_synergy_dataset.R",
        "season_downloaded_at": {season: _iso(seasons[season]) for season in sorted(seasons)},
        "chunk_count": len(chunks),
        "notes": "pulled_at is the newest chunk download. Seasons whose chunks were reused keep their earlier download date.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    args = parser.parse_args()
    record = build_record(args.csv, args.chunks)
    target = args.csv.with_suffix(args.csv.suffix + ".source.json")
    target.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {target.name}: pulled {record['pulled_at']}")


if __name__ == "__main__":
    main()
