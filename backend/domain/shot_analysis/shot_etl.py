from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import re

import numpy as np
import pandas as pd

from domain.shot_analysis.shot_config import normalize_shot_type, season_int_to_str, zone_from_xy

# Paths (relative to backend/)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PBP_DIR = DATA_DIR / "pbp"
SOURCE_PARQUET = PBP_DIR / "nba_pbp_2021_present.parquet"
ALT_SOURCE_PARQUET = DATA_DIR / "nba_pbp_2021_present.parquet"
CLEAN_PARQUET = PBP_DIR / "shots_clean.parquet"

# NBA team id -> abbreviation (common across NBA API datasets)
# Used only as a fallback if the dataset doesn't provide abbreviations.

TEAM_ID_TO_ABBR = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

def _norm_letters(s: str) -> str:
    # Lowercase and keep only letters (removes underscores, spaces, digits, etc.)
    return re.sub(r"[^a-z]+", "", str(s).lower())

def _norm_alnum(s: str) -> str:
    # Lowercase and keep letters+digits (useful for some schemas)
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _pick_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = False,
    prefer: Sequence[str] = (),
) -> Optional[str]:
    """Pick a column from df that matches one of the candidate names."""
    cols = list(df.columns)
    if not cols:
        if required:
            raise ValueError("DataFrame has no columns.")
        return None

    # 1) exact match
    for c in candidates:
        if c in df.columns:
            return c

    # 2) case-insensitive exact match
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]

    # 3) normalized alnum exact match
    alnum_map = {}
    for c in cols:
        k = _norm_alnum(c)
        alnum_map.setdefault(k, c)
    for cand in candidates:
        k = _norm_alnum(cand)
        if k in alnum_map:
            return alnum_map[k]

    # 4) letters-only exact match
    letters_map = {}
    for c in cols:
        k = _norm_letters(c)
        letters_map.setdefault(k, c)
    for cand in candidates:
        k = _norm_letters(cand)
        if k in letters_map:
            return letters_map[k]

    # 5) substring match with scoring
    prefer_lower = [p.lower() for p in prefer]

    def score(colname: str, cand_norm: str) -> tuple[int, int]:
        # Higher is better: prefer tokens, then shorter name
        low = str(colname).lower()
        bonus = sum(1 for p in prefer_lower if p and p in low)
        # prefer shorter names to avoid accidentally selecting verbose unrelated fields
        return (bonus, -len(low))

    for cand in candidates:
        cand_norm = _norm_letters(cand)
        if not cand_norm:
            continue
        matches = [c for c in cols if cand_norm in _norm_letters(c)]
        if matches:
            matches_sorted = sorted(matches, key=lambda c: score(c, cand_norm), reverse=True)
            return matches_sorted[0]

    if required:
        tried = list(candidates)
        sample_cols = cols[:40]
        raise ValueError(
            f"Required column not found. Tried: {tried}\n"
            f"First {len(sample_cols)} columns in parquet: {sample_cols}"
        )
    return None

def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) > 0
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"1", "true", "t", "yes", "y"})

def _coerce_made(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if np.issubdtype(s.dtype, np.number):
        return (s.fillna(0).astype(float) > 0).astype(int)
    s_str = s.astype(str).str.strip().str.lower()
    made = s_str.isin({"1", "true", "t", "made", "make", "good"})
    miss = s_str.isin({"0", "false", "f", "miss", "missed", "no"})
    out = pd.Series(np.where(made, 1, np.where(miss, 0, np.nan)), index=s.index)
    return out.fillna(0).astype(int)

def _parse_clock_to_sec(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return float("nan")
    if ":" in s:
        try:
            mins, secs = s.split(":")
            return float(int(mins) * 60 + int(secs))
        except Exception:
            return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _parse_margin(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper()
    if s in {"", "TIE", "TIED", "0"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return float("nan")

def _compute_distance(x: pd.Series, y: pd.Series) -> pd.Series:
    return np.sqrt(x.astype(float) ** 2 + y.astype(float) ** 2)

def _compute_angle_deg(x: pd.Series, y: pd.Series) -> pd.Series:
    return np.degrees(np.arctan2(y.astype(float), x.astype(float)))

def _normalize_shot_type_from_row(row: pd.Series) -> str:
    return normalize_shot_type(
        type_text=row.get("type_text"),
        type_abbreviation=row.get("type_abbreviation"),
        text=row.get("text"),
    )

def _map_team_id_to_abbr(team_id_series: pd.Series) -> pd.Series:
    ids = pd.to_numeric(team_id_series, errors="coerce")
    mapped = ids.map(TEAM_ID_TO_ABBR)
    # fallback to string form if unknown ids
    out = mapped.where(mapped.notna(), ids.astype("Int64").astype(str))
    out = out.replace({"<NA>": None, "nan": None, "NaN": None})
    return out

def build_shots_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # Identify "shot attempt" rows
    shooting_col = _pick_col(
        df,
        [
            "shooting_play",
            "is_shot_attempt",
            "is_shot",
            "shot_attempt",
            "is_fga",
            "field_goal_attempt",
        ],
        required=False,
    )

    if shooting_col:
        df = df[_coerce_bool(df[shooting_col])].copy()
    else:
        # Fallback for nba_api-style play-by-play: EVENTMSGTYPE in {1,2}
        msgtype_col = _pick_col(
            df,
            ["eventmsgtype", "event_msg_type", "event_type", "eventtype", "event_type_id"],
            required=True,
        )
        msg = pd.to_numeric(df[msgtype_col], errors="coerce")
        df = df[msg.isin([1, 2])].copy()

    # Exclude free throws (hoopR marks them as shooting_play=True)
    _tt_col = _pick_col(df, ["type_text", "shot_type", "event_subtype"], required=False)
    if _tt_col is not None:
        _ft_mask = df[_tt_col].astype(str).str.contains("Free Throw", case=False, na=False)
        if _ft_mask.any():
            df = df[~_ft_mask].copy()
    # Season
    season_col = _pick_col(df, ["season", "season_year", "season_int", "seasonyear"], required=True)
    df["SEASON_STR"] = df[season_col].apply(season_int_to_str)
    # Team abbreviation (robust)
    team_col = _pick_col(
        df,
        [
            "team_abbr",
            "team_abbreviation",
            "team",
            "team_tricode",
            "teamTricode",
            "offense_team",
            "offense_team_abbr",
            "offense_team_abbreviation",
            "possession_team",
            "possession_team_abbr",
            "possession_team_abbreviation",
            "posteam",
            "player1_team_abbreviation",
            "player1_team_abbr",
            "player1_team",
            "player_team_abbreviation",
        ],
        required=False,
        prefer=["player1", "shooter", "offense", "possession", "posteam"],
    )

    if team_col:
        df["TEAM_ABBR"] = df[team_col].astype(str).str.strip()
    else:
        team_id_col = _pick_col(
            df,
            [
                "team_id",
                "teamid",
                "offense_team_id",
                "offenseTeamId",
                "possession_team_id",
                "possessionTeamId",
                "player1_team_id",
                "player_team_id",
            ],
            required=True,
            prefer=["player1", "offense", "possession"],
        )
        df["TEAM_ABBR"] = _map_team_id_to_abbr(df[team_id_col])
    # Home flag
    home_flag_col = _pick_col(df, ["home_flag", "is_home", "team_is_home", "home"], required=False)
    if home_flag_col:
        df["HOME_FLAG"] = _coerce_bool(df[home_flag_col])
    else:
        home_team_col = _pick_col(
            df,
            ["home_team_abbr", "home_team_abbreviation", "home_team_tricode", "homeTeamTricode", "homeTeamAbbreviation"],
            required=False,
        )
        if home_team_col:
            df["HOME_FLAG"] = df[home_team_col].astype(str).str.strip().eq(df["TEAM_ABBR"])
        else:
            df["HOME_FLAG"] = False
    # Opponent abbreviation
    opp_col = _pick_col(
        df,
        [
            "opp_abbr",
            "opponent_team_abbr",
            "opponent_team_abbreviation",
            "opponent",
            "defense_team_abbr",
            "defense_team_abbreviation",
            "def_team",
        ],
        required=False,
        prefer=["defense", "opponent", "visitor", "away"],
    )

    if opp_col:
        df["OPP_ABBR"] = df[opp_col].astype(str).str.strip()
    else:
        home_team_col = _pick_col(
            df,
            ["home_team_abbr", "home_team_abbreviation", "home_team_tricode", "homeTeamTricode", "homeTeamAbbreviation"],
            required=False,
        )
        away_team_col = _pick_col(
            df,
            [
                "away_team_abbr",
                "away_team_abbreviation",
                "visitor_team_abbr",
                "visitor_team_abbreviation",
                "visiting_team_abbr",
                "visitorTeamAbbreviation",
                "awayTeamTricode",
                "visitorTeamTricode",
            ],
            required=False,
        )

        if home_team_col and away_team_col:
            home_abbr = df[home_team_col].astype(str).str.strip()
            away_abbr = df[away_team_col].astype(str).str.strip()
            df["OPP_ABBR"] = np.where(df["HOME_FLAG"], away_abbr, home_abbr)
        else:
            # Try opponent team id mapping
            opp_id_col = _pick_col(
                df,
                ["opponent_team_id", "opp_team_id", "defense_team_id", "opponentTeamId"],
                required=False,
                prefer=["defense", "opponent"],
            )
            if opp_id_col:
                df["OPP_ABBR"] = _map_team_id_to_abbr(df[opp_id_col])
            else:
                df["OPP_ABBR"] = None
    # Shot type normalization
    type_text_col = _pick_col(df, ["type_text", "shot_type", "event_subtype", "eventsubtype"], required=False)
    type_abbrev_col = _pick_col(df, ["type_abbreviation", "type_abbrev", "shot_type_abbr"], required=False)
    text_col = _pick_col(df, ["text", "description", "event_description", "homedescription", "visitordescription"], required=False)

    df["type_text"] = df[type_text_col] if type_text_col else None
    df["type_abbreviation"] = df[type_abbrev_col] if type_abbrev_col else None
    df["text"] = df[text_col] if text_col else None
    df["SHOT_TYPE"] = df.apply(_normalize_shot_type_from_row, axis=1)
    # Coordinates
    x_col = _pick_col(
        df,
        ["x", "x_loc", "loc_x", "coordinate_x", "shot_x", "locx", "shotx", "xcoordinate"],
        required=True,
    )
    y_col = _pick_col(
        df,
        ["y", "y_loc", "loc_y", "coordinate_y", "shot_y", "locy", "shoty", "ycoordinate"],
        required=True,
    )
    df["X"] = pd.to_numeric(df[x_col], errors="coerce")
    df["Y"] = pd.to_numeric(df[y_col], errors="coerce")
    # Shot distance
    dist_col = _pick_col(df, ["shot_distance", "dist", "distance"], required=False)
    if dist_col:
        df["DIST"] = pd.to_numeric(df[dist_col], errors="coerce")
    else:
        # Detect full-court coordinates (hoopR): X spans both halves of court.
        # Basket is at (±41.75, 0).  Compute distance from nearest basket.
        x_range = df["X"].max() - df["X"].min()
        if x_range > 60:  # full-court (94-ft court → range ~90+)
            _BASKET_X = 41.75
            x_abs = df["X"].abs()
            df["DIST"] = np.sqrt((x_abs - _BASKET_X) ** 2 + df["Y"] ** 2)
        else:
            df["DIST"] = _compute_distance(df["X"], df["Y"])

    df["ANGLE"] = _compute_angle_deg(df["X"], df["Y"])
    # Shot value (2 or 3) — avoid generic "value" which substring-matches hoopR's score_value
    shot_value_col = _pick_col(df, ["shot_value", "shot_pts", "shot_value_pts"], required=False)
    is_three_col = _pick_col(df, ["is_three", "is_three_shot", "three_point_attempt"], required=False)

    # hoopR score_value: 0 for misses, 2/3 for makes
    score_value_col = _pick_col(df, ["score_value"], required=False)

    if shot_value_col:
        df["SHOT_VALUE"] = pd.to_numeric(df[shot_value_col], errors="coerce")
    elif is_three_col:
        df["SHOT_VALUE"] = np.where(_coerce_bool(df[is_three_col]), 3, 2)
    elif score_value_col is not None:
        sv = pd.to_numeric(df[score_value_col], errors="coerce").fillna(0).astype(int)
        # For makes (score_value 2 or 3) trust the value; for misses use distance
        df["SHOT_VALUE"] = np.where(
            sv.isin([2, 3]), sv,
            np.where(df["DIST"].fillna(0) >= 22.0, 3, 2)
        )
    else:
        df["SHOT_VALUE"] = np.where(df["DIST"].fillna(0) >= 22.0, 3, 2)
    # Made / Points
    # "scoring_play" is the hoopR column (boolean: TRUE if made)
    made_col = _pick_col(df, ["shot_made", "shot_made_flag", "made", "is_made", "shot_result", "scoring_play"], required=False)
    if made_col:
        df["MADE"] = _coerce_made(df[made_col])
    else:
        # Fallback for nba_api-style: EVENTMSGTYPE 1=made, 2=miss
        msgtype_col = _pick_col(
            df,
            ["eventmsgtype", "event_msg_type", "event_type", "eventtype", "event_type_id"],
            required=False,
        )
        if msgtype_col:
            msg = pd.to_numeric(df[msgtype_col], errors="coerce")
            df["MADE"] = (msg == 1).astype(int)
        else:
            df["MADE"] = 0

    df["POINTS"] = df["SHOT_VALUE"].fillna(0).astype(int) * df["MADE"].fillna(0).astype(int)
    # Period + clock
    period_col = _pick_col(df, ["period", "period_number"], required=False)
    df["PERIOD"] = pd.to_numeric(df[period_col], errors="coerce") if period_col else np.nan

    clock_col = _pick_col(
        df,
        ["clock_sec", "seconds_remaining", "period_time_seconds", "game_clock", "pctimestring", "clock"],
        required=False,
    )
    if clock_col:
        df["CLOCK_SEC"] = df[clock_col].apply(_parse_clock_to_sec)
    else:
        df["CLOCK_SEC"] = np.nan
    # Margin
    margin_col = _pick_col(df, ["score_margin", "margin", "score_diff", "scoremargin"], required=False)
    if margin_col:
        df["MARGIN"] = df[margin_col].apply(_parse_margin)
    else:
        df["MARGIN"] = np.nan
    # IDs
    game_id_col = _pick_col(df, ["game_id", "gameid", "gameId", "GAME_ID"], required=True)
    df["GAME_ID"] = df[game_id_col].astype(str).str.strip()

    shooter_col = _pick_col(
        df,
        [
            "athlete_id_1",
            "shooter_id",
            "player_id",
            "player1_id",
            "player_1_id",
            "person1_id",
            "athlete_id",
            "player1id",
            "PLAYER1_ID",
        ],
        required=False,
        prefer=["player1", "shooter", "athlete1", "person1"],
    )
    df["SHOOTER_ID"] = df[shooter_col].astype(str).str.strip() if shooter_col else None
    # Zone
    df["ZONE"] = [
        zone_from_xy(x, y, d) if np.isfinite(x) and np.isfinite(y) else "unknown"
        for x, y, d in zip(df["X"], df["Y"], df["DIST"])
    ]

    keep_cols = [
        "SEASON_STR",
        "TEAM_ABBR",
        "OPP_ABBR",
        "HOME_FLAG",
        "SHOT_TYPE",
        "SHOT_VALUE",
        "MADE",
        "POINTS",
        "X",
        "Y",
        "DIST",
        "ANGLE",
        "ZONE",
        "PERIOD",
        "CLOCK_SEC",
        "MARGIN",
        "GAME_ID",
        "SHOOTER_ID",
    ]

    clean = df[keep_cols].copy()

    # Drop rows missing core fields
    clean = clean.dropna(subset=["SEASON_STR", "TEAM_ABBR", "GAME_ID", "X", "Y"]).reset_index(drop=True)

    # Deterministic ordering
    sort_cols = [c for c in ["GAME_ID", "PERIOD", "CLOCK_SEC", "TEAM_ABBR"] if c in clean.columns]
    if sort_cols:
        clean = clean.sort_values(sort_cols).reset_index(drop=True)

    return clean

def build_shots_dataset(
    parquet_path: Path = SOURCE_PARQUET,
    output_path: Path = CLEAN_PARQUET,
) -> Path:
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    if not parquet_path.exists():
        if parquet_path == SOURCE_PARQUET and ALT_SOURCE_PARQUET.exists():
            parquet_path = ALT_SOURCE_PARQUET
        else:
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    print(f"[shot_etl] Loading parquet: {parquet_path}")
    raw = pd.read_parquet(parquet_path)
    print(f"[shot_etl] Raw rows: {len(raw):,} | cols: {len(raw.columns)}")

    print("[shot_etl] Building clean shots table...")
    clean = build_shots_clean(raw)
    print(f"[shot_etl] Clean rows: {len(clean):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False)
    print(f"[shot_etl] Saved: {output_path}")

    return output_path

if __name__ == "__main__":
    build_shots_dataset()
