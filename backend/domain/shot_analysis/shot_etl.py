from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import re

import numpy as np
import pandas as pd

from domain.shot_analysis.shot_config import classify_zone, normalize_shot_type, season_int_to_str

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

TEAM_ABBR_ALIASES = {"WSH": "WAS", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "NO": "NOP", "GS": "GSW"}

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

    # Participant and score fields must never be selected by substring.
    # For example, team must not match home_team and home must not match home_score.
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
            return float(int(mins) * 60 + float(secs))
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
    if s in {"TIE", "TIED", "0"}:
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
    # Prefer event participant IDs over any game-level team label.
    team_id_col = _pick_col(df, ["team_id", "player1_team_id", "offense_team_id", "possession_team_id"])
    home_id_col = _pick_col(df, ["home_team_id"])
    away_id_col = _pick_col(df, ["away_team_id", "visitor_team_id"])
    home_col = _pick_col(df, ["home_team_abbr", "home_team_abbrev", "home_team_abbreviation", "homeTeamTricode"])
    away_col = _pick_col(df, ["away_team_abbr", "away_team_abbrev", "away_team_abbreviation", "visitor_team_abbreviation", "awayTeamTricode"])
    if team_id_col and home_id_col and away_id_col and home_col and away_col:
        event_id = pd.to_numeric(df[team_id_col], errors="coerce")
        is_home = event_id.eq(pd.to_numeric(df[home_id_col], errors="coerce"))
        is_away = event_id.eq(pd.to_numeric(df[away_id_col], errors="coerce"))
        if not (is_home ^ is_away).all():
            raise ValueError("A shooting team ID does not match exactly one game participant.")
        df["HOME_FLAG"] = is_home
        df["TEAM_ABBR"] = np.where(is_home, df[home_col], df[away_col])
        df["OPP_ABBR"] = np.where(is_home, df[away_col], df[home_col])
    else:
        team_col = _pick_col(df, ["team_abbr", "team_abbreviation", "team", "offense_team_abbr", "player1_team_abbreviation"])
        if team_id_col:
            df["TEAM_ABBR"] = _map_team_id_to_abbr(df[team_id_col])
        elif team_col:
            df["TEAM_ABBR"] = df[team_col]
        else:
            raise ValueError("A shooting-team participant ID or explicit event team is required.")
        if home_col and away_col:
            df["HOME_FLAG"] = df["TEAM_ABBR"].eq(df[home_col])
            if not (df["TEAM_ABBR"].eq(df[home_col]) | df["TEAM_ABBR"].eq(df[away_col])).all():
                raise ValueError("The shooting team is not a game participant.")
            df["OPP_ABBR"] = np.where(df["HOME_FLAG"], df[away_col], df[home_col])
        else:
            home_flag_col = _pick_col(df, ["home_flag", "is_home", "team_is_home"])
            opp_col = _pick_col(df, ["opp_abbr", "opponent_team_abbreviation", "opponent"])
            df["HOME_FLAG"] = _coerce_bool(df[home_flag_col]) if home_flag_col else None
            df["OPP_ABBR"] = df[opp_col] if opp_col else None
    for name in ["TEAM_ABBR", "OPP_ABBR"]:
        df[name] = df[name].astype("string").str.strip().str.upper().replace(TEAM_ABBR_ALIASES)
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
    raw_x = _pick_col(df, ["coordinate_x_raw"])
    raw_y = _pick_col(df, ["coordinate_y_raw"])
    if raw_x and raw_y:
        lateral = pd.to_numeric(df[raw_x], errors="coerce") - 25.0
        depth = pd.to_numeric(df[raw_y], errors="coerce")
        df["DIST"] = np.hypot(lateral, depth)
        df["ANGLE"] = np.degrees(np.arctan2(lateral, depth))
        zone_x, zone_y = depth, lateral
    else:
        dist_col = _pick_col(df, ["shot_distance", "dist", "distance"])
        df["DIST"] = pd.to_numeric(df[dist_col], errors="coerce") if dist_col else _compute_distance(df["X"], df["Y"])
        df["ANGLE"] = _compute_angle_deg(df["X"], df["Y"])
        zone_x, zone_y = df["X"], df["Y"]

    # Keep the actual attempt value where supplied. A missing miss value is
    # unknown, not a distance-based guess about which side of the line it took.
    value_col = _pick_col(df, ["points_attempted", "shot_value", "shot_pts", "shot_value_pts"])
    attempt_value = pd.to_numeric(df[value_col], errors="coerce") if value_col else pd.Series(np.nan, index=df.index)
    score_value_col = _pick_col(df, ["score_value"])
    scored = pd.to_numeric(df[score_value_col], errors="coerce") if score_value_col else pd.Series(np.nan, index=df.index)
    description = df["text"].fillna("").astype(str)
    explicit_three = description.str.contains(r"three[ -]point|3[ -]point", case=False, regex=True)
    explicit_two = description.str.contains(r"two[ -]point|2[ -]point|layup|dunk|tip shot", case=False, regex=True)
    fallback = pd.Series(np.where(explicit_three, 3.0, np.where(explicit_two, 2.0, np.nan)), index=df.index)
    df["SHOT_VALUE"] = attempt_value.where(attempt_value.isin([2, 3]), scored.where(scored.isin([2, 3]), fallback))
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

    df["POINTS"] = scored.where(scored.isin([0, 2, 3]), df["SHOT_VALUE"] * df["MADE"]).where(df["MADE"].ne(0), 0)
    # Period + clock
    period_col = _pick_col(df, ["period", "period_number"], required=False)
    df["PERIOD"] = pd.to_numeric(df[period_col], errors="coerce") if period_col else np.nan

    clock_col = _pick_col(
        df,
        ["clock_sec", "seconds_remaining", "period_time_seconds", "game_clock", "pctimestring", "clock_display_value", "clock"],
        required=False,
    )
    if clock_col:
        df["CLOCK_SEC"] = df[clock_col].apply(_parse_clock_to_sec)
    else:
        df["CLOCK_SEC"] = np.nan
    # ESPN event scores are after the play. Remove this shot's awarded
    # points before computing the shooting team's score margin.
    home_score = _pick_col(df, ["home_score"])
    away_score = _pick_col(df, ["away_score"])
    margin_col = _pick_col(df, ["score_margin", "margin", "score_diff", "scoremargin"])
    if home_score and away_score and score_value_col:
        home_after = pd.to_numeric(df[home_score], errors="coerce")
        away_after = pd.to_numeric(df[away_score], errors="coerce")
        home_before = home_after - scored.where(df["HOME_FLAG"], 0)
        away_before = away_after - scored.where(~df["HOME_FLAG"], 0)
        df["MARGIN"] = np.where(df["HOME_FLAG"], home_before - away_before, away_before - home_before)
    elif margin_col:
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
    classified = [classify_zone(x, y, d, value) for x, y, d, value in zip(zone_x, zone_y, df["DIST"], df["SHOT_VALUE"])]
    df["ZONE"] = [value[0] for value in classified]
    df["ZONE_METHOD"] = [value[1] for value in classified]
    df["LOCATION_VALID"] = [value[2] for value in classified]

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
        "ZONE_METHOD",
        "LOCATION_VALID",
        "PERIOD",
        "CLOCK_SEC",
        "MARGIN",
        "GAME_ID",
        "SHOOTER_ID",
    ]

    event_id_col = _pick_col(df, ["id", "event_id"])
    if event_id_col:
        df["EVENT_ID"] = df[event_id_col].astype(str)
        keep_cols.append("EVENT_ID")
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
