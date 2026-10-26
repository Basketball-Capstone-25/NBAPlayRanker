"""Read-only audit of the pinned raw events and current shot files.

Run from the repository root with backend/.venv/bin/python and pass raw-dir.
Writes only sibling audit JSON; does not call the app or train a model.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
RAW = Path(sys.argv[1])
MANIFEST = json.loads((ROOT / "docs/iteration-17/shot-repair/source-manifest.json").read_text())
OLD_PROOF = json.loads((ROOT / "docs/iteration-17/shot-repair/rebuild-validation.json").read_text())
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "backend/data/pbp"
OUTPUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "raw-event-audit-before.json"
clean = pd.read_parquet(DATA / "shots_clean.parquet")
ALIASES = {"WSH": "WAS", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "NO": "NOP", "GS": "GSW"}
teams = set(clean.TEAM_ABBR)
summaries, anomalies, samples = [], [], []


def records(frame):
    return json.loads(frame.to_json(orient="records"))


for asset in MANIFEST:
    path = RAW / asset["name"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == asset["sha256"]
    cols = ["id", "game_id", "game_play_number", "season", "season_type", "shooting_play", "scoring_play", "score_value", "text", "type_text", "team_id", "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev", "home_score", "away_score", "coordinate_x_raw", "coordinate_y_raw", "coordinate_x", "coordinate_y", "points_attempted", "period_number", "clock_display_value"]
    schema = pq.read_schema(path).names
    raw = pd.read_parquet(path, columns=[c for c in cols if c in schema])
    home = raw.home_team_abbrev.replace(ALIASES)
    away = raw.away_team_abbrev.replace(ALIASES)
    eligible = raw.season_type.isin([2, 3]) & home.isin(teams) & away.isin(teams)
    raw = raw.loc[eligible].copy()
    # Compare score margin with the preceding event's score, independently of
    # subtracting the current shot's points from its post-event scoreboard.
    chronological = raw.sort_values(["game_id", "game_play_number"])
    chronological["prior_home"] = chronological.groupby("game_id").home_score.shift().fillna(0)
    chronological["prior_away"] = chronological.groupby("game_id").away_score.shift().fillna(0)
    shots = chronological.loc[chronological.shooting_play & ~chronological.type_text.str.contains("Free Throw", case=False, na=False)].copy()
    shots["EVENT_ID"] = shots.id.astype(str)
    shots = shots.set_index("EVENT_ID")
    season = f"{int(raw.season.iloc[0])-1}-{int(raw.season.iloc[0])%100:02}"
    current = clean.loc[clean.SEASON_STR.eq(season)].set_index("EVENT_ID")
    missing = shots.index.difference(current.index)
    extra = current.index.difference(shots.index)
    assert len(extra) == 0
    joined = current.join(shots, how="left")
    is_home = joined.team_id.eq(joined.home_team_id)
    is_away = joined.team_id.eq(joined.away_team_id)
    expected_team = pd.Series(np.where(is_home, joined.home_team_abbrev, joined.away_team_abbrev), index=joined.index).replace(ALIASES)
    expected_opp = pd.Series(np.where(is_home, joined.away_team_abbrev, joined.home_team_abbrev), index=joined.index).replace(ALIASES)
    lateral = joined.coordinate_x_raw - 25
    depth = joined.coordinate_y_raw
    sign = np.where(is_home, -1., 1.)
    expected_x = (depth - 41.75) * sign
    expected_y = lateral * sign
    expected_margin = np.where(is_home, joined.home_score-joined.away_score, joined.away_score-joined.home_score) - joined.score_value
    prior_margin = np.where(is_home, joined.prior_home-joined.prior_away, joined.prior_away-joined.prior_home)
    mismatch_prior = ~np.isclose(joined.MARGIN, prior_margin)
    # NBA arc/corner geometry used here is independent of the application's
    # current 22-ft radial cutoff and 14-ft corner-depth cutoff.
    corner_depth = np.sqrt(23.75**2 - 22**2)
    geometric_three = ((depth <= corner_depth) & (abs(lateral) > 22)) | ((depth > corner_depth) & (np.hypot(lateral, depth) > 23.75))
    labelled_three = joined.ZONE.isin(["arc3", "corner3"])
    known_two_wrong = joined.SHOT_VALUE.eq(2) & labelled_three
    known_three_wrong = joined.SHOT_VALUE.eq(3) & joined.ZONE.isin(["rim", "paint", "mid"])
    wrong_corner = joined.ZONE.eq("corner3") & depth.gt(corner_depth)
    declared_value = pd.to_numeric(joined.get("points_attempted", pd.Series(np.nan, index=joined.index)), errors="coerce")
    checks = {
        "raw_sha256_matches": digest == asset["sha256"],
        "events_present_in_raw": int(joined.id.notna().sum()),
        "unique_event_ids": current.index.is_unique,
        "invalid_participant_id_rows": int((~(is_home ^ is_away)).sum()),
        "team_mismatches": int(joined.TEAM_ABBR.ne(expected_team).sum()),
        "opponent_mismatches": int(joined.OPP_ABBR.ne(expected_opp).sum()),
        "home_flag_mismatches": int(joined.HOME_FLAG.ne(is_home).sum()),
        "point_mismatches": int(joined.POINTS.ne(joined.score_value).sum()),
        "made_mismatches": int(joined.MADE.ne(joined.scoring_play.astype(int)).sum()),
        "supplied_attempt_value_mismatches": int((declared_value.isin([2,3]) & joined.SHOT_VALUE.ne(declared_value)).sum()),
        "post_event_adjusted_margin_mismatches": int((~np.isclose(joined.MARGIN, expected_margin)).sum()),
        "preceding_score_margin_disagreements": int(mismatch_prior.sum()),
        "either_scoreboard_transition_disagreements": int((~np.isclose(joined.home_score-joined.prior_home, joined.score_value.where(is_home,0)) | ~np.isclose(joined.away_score-joined.prior_away, joined.score_value.where(~is_home,0))).sum()),
        "x_transform_mismatches": int((~np.isclose(joined.X, expected_x)).sum()),
        "y_transform_mismatches": int((~np.isclose(joined.Y, expected_y)).sum()),
        "distance_feet_mismatches": int((~np.isclose(joined.DIST, np.hypot(lateral, depth))).sum()),
        "angle_degrees_mismatches": int((~np.isclose(joined.ANGLE, np.degrees(np.arctan2(lateral, depth)))).sum()),
        "known_twos_in_three_point_zones": int(known_two_wrong.sum()),
        "known_threes_in_two_point_zones": int(known_three_wrong.sum()),
        "current_corner_beyond_corner_join": int(wrong_corner.sum()),
        "three_area_geometry_disagreements": int(labelled_three.ne(geometric_three).sum()),
        "known_values_disagree_with_rounded_geometry": int((joined.SHOT_VALUE.isin([2,3]) & joined.SHOT_VALUE.eq(3).ne(geometric_three)).sum()),
        "missing_coordinate_raw_rows": int(joined[["coordinate_x_raw","coordinate_y_raw"]].isna().any(axis=1).sum()),
        "flagged_invalid_locations": int(joined.LOCATION_VALID.eq(False).sum()) if "LOCATION_VALID" in joined else None,
        "shots_outside_physical_court_coordinates": int(((abs(lateral)>25) | (depth < -5.25) | (depth > 88.75)).sum()),
    }
    old_samples = [s for s in OLD_PROOF["spot_checks"] if s["season"] == season]
    proof_mismatches = 0
    proof_quarantined = 0
    for old in old_samples:
        if old["event_id"] not in joined.index:
            proof_quarantined += 1
            continue
        event = joined.loc[old["event_id"]]
        proof_mismatches += int(not(event.TEAM_ABBR == old["team"] and event.OPP_ABBR == old["opponent"] and event.MARGIN == old["margin_before_shot"] and event.POINTS == old["awarded_points"] and event.text == old["source_text"]))
    checks["existing_100_sample_proof_mismatches"] = proof_mismatches
    checks["existing_samples_in_quarantined_games"] = proof_quarantined
    sample = joined.sample(n=100, random_state=20261025).reset_index()
    keep = ["EVENT_ID","SEASON_STR","GAME_ID","TEAM_ABBR","OPP_ABBR","HOME_FLAG","text","SHOT_VALUE","MADE","POINTS","MARGIN","X","Y","DIST","ZONE","coordinate_x_raw","coordinate_y_raw"]
    samples.extend(records(sample[keep]))
    for kind,mask in [("known_two_in_three_zone",known_two_wrong),("known_three_in_two_zone",known_three_wrong),("scoreboard_sequence_disagreement",mismatch_prior),("corner_beyond_join",wrong_corner)]:
        example = joined.loc[mask].head(4).reset_index()
        anomalies.extend({"finding":kind,**r} for r in records(example[keep+["prior_home","prior_away","home_score","away_score"]]))
    summaries.append({"season":season,"source":asset["name"],"sha256":digest,"rows":len(current),"raw_shots_missing_from_clean":len(missing),"games":int(current.GAME_ID.nunique()),"teams":int(current.TEAM_ABBR.nunique()),"checks":checks})
    print(json.dumps(summaries[-1]),flush=True)

report = {"checked_at_utc":datetime.now(timezone.utc).isoformat(),"clean_sha256":hashlib.sha256((DATA/'shots_clean.parquet').read_bytes()).hexdigest(),"read_only":True,"rows":len(clean),"seasons":summaries,"new_sample_seed":20261025,"new_samples":samples,"representative_findings":anomalies,"coordinate_contract":"Provider raw coordinates are feet with basket at (25,0); transformed full-court coordinates have midcourt origin and are rotated for home plays.","zone_geometry_source":"https://cdn.nba.com/manage/2026/01/Official-2025-26-NBA-Playing-Rules.pdf","coordinate_source":"https://py.sportsdataverse.org/changelog-unreleased","limitations":["Raw provider coordinates are rounded and may disagree with explicit attempted shot value near the line.","This audit compares the pinned provider snapshot; it does not independently establish every provider event's real-world accuracy."]}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/OUTPUT_NAME).write_text(json.dumps(report,indent=2)+'\n')
