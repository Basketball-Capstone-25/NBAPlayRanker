"""Check the baseline's offence/defence weight against team-game shot results.

Run from the repository root:
    python backend/data/etl/check_offense_defense_weight.py

Writes docs/model/offense-defense-weight.json. The latest season is left out
when it has fewer than half the team-games of a typical season, because a
season in progress has unstable season averages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import pandas as pd

from domain.statistical_analysis.offense_defense_weight import estimate_offense_share

DEFAULT_SHOTS = BACKEND / "data" / "pbp" / "shots_clean.parquet"
DEFAULT_OUT = ROOT / "docs" / "model" / "offense-defense-weight.json"


def complete_seasons(shots: pd.DataFrame) -> list[str]:
    """Seasons with at least half the team-games of the median season."""
    games = shots.groupby("SEASON_STR")["GAME_ID"].nunique()
    return sorted(games[games >= 0.5 * games.median()].index.tolist())


def main() -> None:
    """Run the check and save the result with the source file's hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    shots = pd.read_parquet(args.shots, columns=["SEASON_STR", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", "POINTS"])
    seasons = complete_seasons(shots)
    result = estimate_offense_share(shots, seasons=seasons)
    result.update({
        "source": str(args.shots.relative_to(ROOT)) if args.shots.is_relative_to(ROOT) else str(args.shots),
        "source_sha256": hashlib.sha256(args.shots.read_bytes()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_default_w_off": 0.5,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Offence share {result['offense_share']:.3f} "
          f"(95% {result['offense_share_95'][0]:.3f} to {result['offense_share_95'][1]:.3f}) "
          f"from {result['team_games']} team-games. Saved {args.out}")


if __name__ == "__main__":
    main()
