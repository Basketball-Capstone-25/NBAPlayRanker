from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import re

import numpy as np
import pandas as pd

from shot_config import normalize_shot_type, season_int_to_str, zone_from_xy

# ---------------------------------------------------------------------
# Paths (relative to backend/)
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
PBP_DIR = DATA_DIR / "pbp"
SOURCE_PARQUET = PBP_DIR / "nba_pbp_2021_present.parquet"
ALT_SOURCE_PARQUET = DATA_DIR / "nba_pbp_2021_present.parquet"
CLEAN_PARQUET = PBP_DIR / "shots_clean.parquet"

# ---------------------------------------------------------------------
# NBA team id -> abbreviation (common across NBA API datasets)
# Used only as a fallback if the dataset doesn't provide abbreviations.
# ---------------------------------------------------------------------

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

# ESPN-style parquet ids seen in your raw file.
# These are small integer team ids, unlike the NBA API 1610612xxx ids above.
ESPN_TEAM_ID_TO_ABBR = {
    1: "ATL",
    2: "BOS",
    3: "NOP",
    4: "CHI",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GSW",
    10: "HOU",
    11: "IND",
    12: "LAC",
    13: "LAL",
    14: "MIA",
    15: "MIL",
    16: "MIN",
    17: "BKN",
    18: "NYK",
    19: "ORL",
    20: "PHI",
    21: "PHX",
    22: "POR",
    23: "SAC",
    24: "SAS",
    25: "OKC",
    26: "UTA",
    27: "WAS",
    28: "TOR",
    29: "MEM",
    30: "CHA",
}

TEAM_ID_TO_ABBR_ALL = {**TEAM_ID_TO_ABBR, **ESPN_TEAM_ID_TO_ABBR}


def _norm_letters(s: str) -> str:
    return re.sub(r"[^a-z]+", "", str(s).lower())


def _norm_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _pick_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = False,
    prefer: Sequence[str] = (),
) -> Optional[str]:
    """
    Pick a column from df that matches one of the candidate names.

    Matching strategy (in order):
    1) Exact match
    2) Case-insensitive exact match
    3) Normalized alnum exact match
    4) Normalized letters-only exact match
    5) Normalized substring match (best-scored among matches)
    """
    cols = list(df.columns)
    if not cols:
        if required:
            raise ValueError("DataFrame has no columns.")
        return None

    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]

    alnum_map = {}
    for c in cols:
        k = _norm_alnum(c)
        alnum_map.setdefault(k, c)
    for cand in candidates:
        k = _norm_alnum(cand)
        if k in alnum_map:
            return alnum_map[k]

    letters_map = {}
    for c in cols:
        k = _norm_letters(c)
        letters_map.setdefault(k, c)
    for cand in candidates:
        k = _norm_letters(cand)
        if k in letters_map:
            return letters_map[k]

    prefer_lower = [p.lower() for p in prefer]

    def score(colname: str, cand_norm: str) -> tuple[int, int]:
        low = str(colname).lower()
        bonus = sum(1 for p in prefer_lower if p and p in low)
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
    made = s_str.isin({"1", "true", "t", "made", "make", "good", "makes"})
    miss = s_str.isin({"0", "false", "f", "miss", "missed", "no", "misses"})
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
    type_text = row.get("type_text")
    type_abbreviation = row.get("type_abbreviation")
    text = row.get("text")

    raw = " ".join([str(v) for v in [type_text, type_abbreviation, text] if v is not None]).lower()
    if "free throw" in raw:
        return "Free Throw"
    if "tip in" in raw or "tip shot" in raw:
        return "Tip Shot"

    label = normalize_shot_type(
        type_text=type_text,
        type_abbreviation=type_abbreviation,
        text=text,
    )
    return label


def _map_team_id_to_abbr(team_id_series: pd.Series) -> pd.Series:
    ids = pd.to_numeric(team_id_series, errors="coerce")
    mapped = ids.map(TEAM_ID_TO_ABBR_ALL)
    out = mapped.where(mapped.notna(), ids.astype("Int64").astype(str))
    out = out.replace({"<NA>": None, "nan": None, "NaN": None})
    return out


def _infer_shot_value(
    *,
    score_value: pd.Series | None,
    points_attempted: pd.Series | None,
    type_text: pd.Series | None,
    text: pd.Series | None,
    dist: pd.Series,
) -> pd.Series:
    idx = dist.index
    out = pd.Series(np.nan, index=idx, dtype=float)

    if points_attempted is not None:
        pts = pd.to_numeric(points_attempted, errors="coerce")
        out = out.where(out.notna(), pts)

    if score_value is not None:
        sv = pd.to_numeric(score_value, errors="coerce")
        out = out.where(out.notna(), sv.where(sv > 0))

    raw_text = pd.Series("", index=idx, dtype=object)
    if type_text is not None:
        raw_text = raw_text.str.cat(type_text.fillna("").astype(str), sep=" ")
    if text is not None:
        raw_text = raw_text.str.cat(text.fillna("").astype(str), sep=" ")
    s = raw_text.str.lower()

    out = np.where(s.str.contains("free throw", na=False), 1, out)
    out = np.where(s.str.contains("three point|3-point|3pt|3-pt", regex=True, na=False), 3, out)
    out = np.where(
        s.str.contains("two point|2-point|2pt|2-pt|layup|dunk|hook shot|tip shot|tip in", regex=True, na=False),
        np.where(pd.isna(out), 2, out),
        out,
    )

    dist_f = pd.to_numeric(dist, errors="coerce")
    out = np.where(pd.isna(out) & dist_f.notna() & (dist_f >= 22.0), 3, out)
    out = np.where(pd.isna(out) & dist_f.notna(), 2, out)
    out = pd.Series(out, index=idx)
    return out.fillna(2).clip(lower=1, upper=3).astype(int)


def _infer_made(
    *,
    made_raw: pd.Series | None,
    scoring_play: pd.Series | None,
    score_value: pd.Series | None,
    text: pd.Series | None,
) -> pd.Series:
    idx = text.index if text is not None else (made_raw.index if made_raw is not None else scoring_play.index if scoring_play is not None else score_value.index)
    out = pd.Series(np.nan, index=idx, dtype=float)

    if made_raw is not None:
        vals = _coerce_made(made_raw)
        out = out.where(out.notna(), vals)

    if scoring_play is not None:
        vals = _coerce_bool(scoring_play).astype(int)
        out = out.where(out.notna(), vals)

    if score_value is not None:
        vals = (pd.to_numeric(score_value, errors="coerce").fillna(0) > 0).astype(int)
        out = out.where(out.notna(), vals)

    if text is not None:
        s = text.fillna("").astype(str).str.lower()
        makes = s.str.contains(r"\bmakes\b|\bmade\b", regex=True, na=False)
        misses = s.str.contains(r"\bmisses\b|\bmissed\b|\bblocks\b|\bblocked\b", regex=True, na=False)
        out = np.where(makes, 1, out)
        out = np.where(misses & pd.isna(out), 0, out)
        out = pd.Series(out, index=idx)

    return out.fillna(0).astype(int)


def build_shots_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

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
        msgtype_col = _pick_col(
            df,
            ["eventmsgtype", "event_msg_type", "event_type", "eventtype", "event_type_id"],
            required=True,
        )
        msg = pd.to_numeric(df[msgtype_col], errors="coerce")
        df = df[msg.isin([1, 2])].copy()

    season_col = _pick_col(df, ["season", "season_year", "season_int", "seasonyear"], required=True)
    df["SEASON_STR"] = df[season_col].apply(season_int_to_str)

    team_col = _pick_col(
        df,
        [
            "team_abbr",
            "team_abbreviation",
            "team_tricode",
            "offense_team_abbr",
            "offense_team_abbreviation",
            "possession_team_abbr",
            "possession_team_abbreviation",
            "posteam",
            "player1_team_abbreviation",
            "player1_team_abbr",
            "player_team_abbreviation",
        ],
        required=False,
        prefer=["player1", "shooter", "offense", "possession", "posteam"],
    )

    home_team_col = _pick_col(
        df,
        ["home_team_abbr", "home_team_abbreviation", "home_team_abbrev", "homeTeamTricode", "homeTeamAbbreviation"],
        required=False,
    )
    away_team_col = _pick_col(
        df,
        [
            "away_team_abbr",
            "away_team_abbreviation",
            "away_team_abbrev",
            "visitor_team_abbr",
            "visitor_team_abbreviation",
            "visiting_team_abbr",
            "visitorTeamAbbreviation",
            "awayTeamTricode",
            "visitorTeamTricode",
        ],
        required=False,
    )

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
        required=False,
        prefer=["player1", "offense", "possession", "team"],
    )
    home_team_id_col = _pick_col(df, ["home_team_id", "hometeamid"], required=False)
    away_team_id_col = _pick_col(df, ["away_team_id", "visitorteamid", "awayteamid"], required=False)

    if team_col:
        df["TEAM_ABBR"] = df[team_col].astype(str).str.strip()
    elif team_id_col and home_team_id_col and away_team_id_col and home_team_col and away_team_col:
        team_ids = pd.to_numeric(df[team_id_col], errors="coerce")
        home_ids = pd.to_numeric(df[home_team_id_col], errors="coerce")
        away_ids = pd.to_numeric(df[away_team_id_col], errors="coerce")
        home_abbr = df[home_team_col].astype(str).str.strip()
        away_abbr = df[away_team_col].astype(str).str.strip()
        df["TEAM_ABBR"] = np.where(team_ids.eq(home_ids), home_abbr, np.where(team_ids.eq(away_ids), away_abbr, _map_team_id_to_abbr(df[team_id_col])))
    elif team_id_col:
        df["TEAM_ABBR"] = _map_team_id_to_abbr(df[team_id_col])
    else:
        raise ValueError("Could not determine shooting team abbreviation from parquet columns.")

    home_flag_col = _pick_col(df, ["home_flag", "is_home", "team_is_home", "home"], required=False)
    if home_flag_col:
        df["HOME_FLAG"] = _coerce_bool(df[home_flag_col])
    elif home_team_col:
        df["HOME_FLAG"] = df[home_team_col].astype(str).str.strip().eq(df["TEAM_ABBR"])
    elif team_id_col and home_team_id_col:
        team_ids = pd.to_numeric(df[team_id_col], errors="coerce")
        home_ids = pd.to_numeric(df[home_team_id_col], errors="coerce")
        df["HOME_FLAG"] = team_ids.eq(home_ids)
    else:
        df["HOME_FLAG"] = False

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
    elif home_team_col and away_team_col:
        home_abbr = df[home_team_col].astype(str).str.strip()
        away_abbr = df[away_team_col].astype(str).str.strip()
        df["OPP_ABBR"] = np.where(df["HOME_FLAG"], away_abbr, home_abbr)
    else:
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

    type_text_col = _pick_col(df, ["type_text", "shot_type", "event_subtype", "eventsubtype"], required=False)
    type_abbrev_col = _pick_col(df, ["type_abbreviation", "type_abbrev", "shot_type_abbr"], required=False)
    text_col = _pick_col(df, ["text", "description", "event_description", "homedescription", "visitordescription"], required=False)

    df["type_text"] = df[type_text_col] if type_text_col else None
    df["type_abbreviation"] = df[type_abbrev_col] if type_abbrev_col else None
    df["text"] = df[text_col] if text_col else None
    df["SHOT_TYPE"] = df.apply(_normalize_shot_type_from_row, axis=1)

    x_col = _pick_col(
        df,
        ["coordinate_x", "x", "x_loc", "loc_x", "coordinate_x_raw", "shot_x", "locx", "shotx", "xcoordinate"],
        required=True,
    )
    y_col = _pick_col(
        df,
        ["coordinate_y", "y", "y_loc", "loc_y", "coordinate_y_raw", "shot_y", "locy", "shoty", "ycoordinate"],
        required=True,
    )
    df["X"] = pd.to_numeric(df[x_col], errors="coerce")
    df["Y"] = pd.to_numeric(df[y_col], errors="coerce")
    df.loc[df["X"].abs() > 100, "X"] = np.nan
    df.loc[df["Y"].abs() > 100, "Y"] = np.nan

    dist_col = _pick_col(df, ["shot_distance", "dist", "distance"], required=False)
    if dist_col:
        df["DIST"] = pd.to_numeric(df[dist_col], errors="coerce")
    else:
        df["DIST"] = _compute_distance(df["X"], df["Y"])

    df["ANGLE"] = _compute_angle_deg(df["X"], df["Y"])

    score_value_col = _pick_col(df, ["score_value", "shot_value", "shot_pts", "value", "shot_value_pts"], required=False)
    points_attempted_col = _pick_col(df, ["points_attempted", "shot_points_attempted"], required=False)
    df["SHOT_VALUE"] = _infer_shot_value(
        score_value=df[score_value_col] if score_value_col else None,
        points_attempted=df[points_attempted_col] if points_attempted_col else None,
        type_text=df["type_text"],
        text=df["text"],
        dist=df["DIST"],
    )

    made_col = _pick_col(df, ["shot_made", "shot_made_flag", "made", "is_made", "shot_result"], required=False)
    scoring_play_col = _pick_col(df, ["scoring_play", "is_scoring_play"], required=False)
    df["MADE"] = _infer_made(
        made_raw=df[made_col] if made_col else None,
        scoring_play=df[scoring_play_col] if scoring_play_col else None,
        score_value=df[score_value_col] if score_value_col else None,
        text=df["text"],
    )
    df["POINTS"] = df["SHOT_VALUE"].fillna(0).astype(int) * df["MADE"].fillna(0).astype(int)

    period_col = _pick_col(df, ["period", "period_number", "qtr"], required=False)
    df["PERIOD"] = pd.to_numeric(df[period_col], errors="coerce") if period_col else np.nan

    clock_col = _pick_col(
        df,
        [
            "time",
            "clock_display_value",
            "pctimestring",
            "game_clock",
            "end_quarter_seconds_remaining",
            "start_quarter_seconds_remaining",
            "seconds_remaining",
            "period_time_seconds",
            "clock_sec",
            "clock",
        ],
        required=False,
    )
    if clock_col:
        df["CLOCK_SEC"] = df[clock_col].apply(_parse_clock_to_sec)
    else:
        minutes_col = _pick_col(df, ["clock_minutes"], required=False)
        seconds_col = _pick_col(df, ["clock_seconds"], required=False)
        if minutes_col and seconds_col:
            mins = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0)
            secs = pd.to_numeric(df[seconds_col], errors="coerce").fillna(0)
            df["CLOCK_SEC"] = mins * 60 + secs
        else:
            df["CLOCK_SEC"] = np.nan

    margin_col = _pick_col(df, ["score_margin", "margin", "score_diff", "scoremargin"], required=False)
    if margin_col:
        df["MARGIN"] = df[margin_col].apply(_parse_margin)
    else:
        home_score_col = _pick_col(df, ["home_score"], required=False)
        away_score_col = _pick_col(df, ["away_score"], required=False)
        if home_score_col and away_score_col:
            home_score = pd.to_numeric(df[home_score_col], errors="coerce")
            away_score = pd.to_numeric(df[away_score_col], errors="coerce")
            df["MARGIN"] = np.where(df["HOME_FLAG"], home_score - away_score, away_score - home_score)
        else:
            df["MARGIN"] = np.nan

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

    df["ZONE"] = [
        zone_from_xy(x, y, d) if np.isfinite(x) and np.isfinite(y) and np.isfinite(d) else "unknown"
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
    clean = clean.dropna(subset=["SEASON_STR", "TEAM_ABBR", "GAME_ID", "X", "Y"]).reset_index(drop=True)

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
    print(f"[shot_etl] Makes: {int(clean['MADE'].sum()):,} | Points: {int(clean['POINTS'].sum()):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False)
    print(f"[shot_etl] Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    build_shots_dataset()