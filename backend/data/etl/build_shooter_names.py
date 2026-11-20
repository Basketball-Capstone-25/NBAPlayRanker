"""Write backend/data/pbp/shooter_names.csv from the raw play-by-play file (SCRUM-825).

Run once from the repository root after the raw file changes:

    python backend/data/etl/build_shooter_names.py

Only the ID and play-text columns are read. The cleaned shots are not changed.
"""

from __future__ import annotations

from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import pandas as pd
import pyarrow.parquet as pq

from domain.shot_analysis.shooter_names import shooter_names

RAW = BACKEND / "data" / "pbp" / "nba_pbp_2021_present.parquet"
OUT = BACKEND / "data" / "pbp" / "shooter_names.csv"
ID_COLUMNS = ["athlete_id_1", "shooter_id", "player_id", "player1_id", "athlete_id"]
TEXT_COLUMNS = ["text", "description", "event_description"]


def main() -> None:
    names = pq.read_schema(RAW).names
    id_col = next((c for c in ID_COLUMNS if c in names), None)
    text_col = next((c for c in TEXT_COLUMNS if c in names), None)
    if not id_col or not text_col:
        raise SystemExit(f"{RAW.name} has no shooter ID or play text column.")
    rows = pd.read_parquet(RAW, columns=[id_col, text_col])
    table = shooter_names(rows, id_col, text_col)
    table.to_csv(OUT, index=False)
    print(f"Wrote {len(table):,} player names to {OUT.relative_to(BACKEND.parent)} from {len(rows):,} plays.")


if __name__ == "__main__":
    main()
