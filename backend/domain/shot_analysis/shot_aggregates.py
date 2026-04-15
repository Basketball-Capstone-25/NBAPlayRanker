from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from domain.shot_analysis.shot_etl import CLEAN_PARQUET

# Paths

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PBP_DIR = DATA_DIR / "pbp"
AGG_PARQUET = PBP_DIR / "shots_agg.parquet"

def _reliability_weight(attempts: pd.Series) -> pd.Series:
    max_log = np.log1p(attempts).max()
    if max_log <= 0:
        return pd.Series(np.zeros(len(attempts)), index=attempts.index)
    return np.log1p(attempts) / max_log

def _agg_stats(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), as_index=False).agg(
        attempts=("MADE", "size"),
        makes=("MADE", "sum"),
        points=("POINTS", "sum"),
    )
    grouped["EPA"] = np.where(grouped["attempts"] > 0, grouped["points"] / grouped["attempts"], np.nan)
    grouped["RELIABILITY_WEIGHT"] = _reliability_weight(grouped["attempts"])
    return grouped

def _build_level(
    df: pd.DataFrame,
    *,
    level: str,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    out = _agg_stats(df, group_cols=group_cols)
    out["LEVEL"] = level
    return out

def _prep_offense_defense(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    off = df.copy()
    off["ROLE"] = "offense"

    # Defense allowed: swap TEAM_ABBR to opponent, if available
    if "OPP_ABBR" in df.columns:
        deff = df.copy()
        deff["TEAM_ABBR"] = deff["OPP_ABBR"]
        deff["ROLE"] = "defense_allowed"
    else:
        deff = pd.DataFrame(columns=df.columns.tolist() + ["ROLE"])

    return off, deff

def build_shot_aggregates(clean_df: pd.DataFrame) -> pd.DataFrame:
    off, deff = _prep_offense_defense(clean_df)
    all_df = pd.concat([off, deff], ignore_index=True)

    base_cols = ["ROLE", "SEASON_STR", "TEAM_ABBR"]

    by_shot_type = _build_level(
        all_df,
        level="shot_type",
        group_cols=base_cols + ["SHOT_TYPE"],
    )
    by_zone = _build_level(
        all_df,
        level="zone",
        group_cols=base_cols + ["ZONE"],
    )
    by_shot_type_zone = _build_level(
        all_df,
        level="shot_type_zone",
        group_cols=base_cols + ["SHOT_TYPE", "ZONE"],
    )

    agg = pd.concat([by_shot_type, by_zone, by_shot_type_zone], ignore_index=True)
    return agg

def build_league_baselines(agg_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ROLE", "LEVEL", "SEASON_STR"]
    keys = ["SHOT_TYPE", "ZONE"]

    def _league(group: pd.DataFrame) -> pd.Series:
        attempts = group["attempts"].sum()
        points = group["points"].sum()
        epa = points / attempts if attempts > 0 else np.nan
        return pd.Series(
            {
                "LEAGUE_ATTEMPTS": attempts,
                "LEAGUE_POINTS": points,
                "EPA_LEAGUE": epa,
            }
        )

    league = (
        agg_df.groupby(group_cols + keys, as_index=False)
        .apply(_league)
        .reset_index(drop=True)
    )
    league["LEAGUE_REL_WEIGHT"] = _reliability_weight(league["LEAGUE_ATTEMPTS"])
    return league

def build_and_save_aggregates(
    *,
    clean_path: Path = CLEAN_PARQUET,
    output_path: Path = AGG_PARQUET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_path = Path(clean_path)
    if not clean_path.exists():
        raise FileNotFoundError(f"shots_clean.parquet not found: {clean_path}")

    print(f"[shot_aggregates] Loading clean shots: {clean_path}")
    clean = pd.read_parquet(clean_path)
    print(f"[shot_aggregates] Clean rows: {len(clean):,}")

    print("[shot_aggregates] Building aggregates...")
    agg = build_shot_aggregates(clean)
    print(f"[shot_aggregates] Agg rows: {len(agg):,}")

    print("[shot_aggregates] Building league baselines...")
    league = build_league_baselines(agg)
    print(f"[shot_aggregates] League rows: {len(league):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(output_path, index=False)
    print(f"[shot_aggregates] Saved aggregates: {output_path}")

    league_path = output_path.with_name("shots_agg_league.parquet")
    league.to_parquet(league_path, index=False)
    print(f"[shot_aggregates] Saved league baselines: {league_path}")

    return agg, league

if __name__ == "__main__":
    build_and_save_aggregates()
