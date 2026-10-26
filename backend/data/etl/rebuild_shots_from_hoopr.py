"""Rebuild into a staging folder from the six pinned hoopR Parquet downloads.

Does not replace runtime files. Review validation.json before promoting results.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from domain.shot_analysis.shot_etl import build_shots_clean, TEAM_ID_TO_ABBR, TEAM_ABBR_ALIASES
from domain.shot_analysis.shot_quality import inspect_shot_rows
from domain.shot_analysis.shot_source_quality import quarantine_score_discontinuities
from domain.shot_analysis.shot_ml_models import get_feature_spec
from domain.shot_analysis.shot_aggregates import build_and_save_aggregates
from infrastructure.data_access.pbp_clean import build_canonical_from_clean


def rebuild(raw_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_dir / "download-manifest.json").read_text())
    frames, summaries, samples, quarantines = [], [], [], []
    for asset in sorted(manifest, key=lambda item: item["name"]):
        source = raw_dir / asset["name"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Source hash mismatch: {source.name}")
        raw = pd.read_parquet(source)
        # Regular season and playoffs, with actual NBA team participants.
        home = raw.home_team_abbrev.replace(TEAM_ABBR_ALIASES)
        away = raw.away_team_abbrev.replace(TEAM_ABBR_ALIASES)
        keep = raw.season_type.isin([2, 3]) & home.isin(TEAM_ID_TO_ABBR.values()) & away.isin(TEAM_ID_TO_ABBR.values())
        eligible = raw[keep].copy()
        eligible, quarantine = quarantine_score_discontinuities(eligible)
        quarantines.append({"source": source.name, **quarantine})
        clean = build_shots_clean(eligible)
        if clean.EVENT_ID.duplicated().any():
            raise ValueError(f"Duplicate event IDs: {source.name}")
        quality = inspect_shot_rows(clean)
        if not quality["valid_attribution"]:
            raise ValueError(f"Invalid participants in {source.name}: {quality}")
        features = get_feature_spec()["numeric_features"]
        incomplete = {name: int((~np.isfinite(clean[name])).sum()) for name in features if not np.isfinite(clean[name]).all()}
        if incomplete:
            raise ValueError(f"Incomplete model features in {source.name}: {incomplete}")
        lookup = eligible.assign(EVENT_ID=eligible.id.astype(str)).set_index("EVENT_ID")
        sample = clean.sample(n=min(100, len(clean)), random_state=42)
        for _, shot in sample.iterrows():
            event = lookup.loc[shot.EVENT_ID]
            is_home = event.team_id == event.home_team_id
            team = TEAM_ABBR_ALIASES.get(event.home_team_abbrev if is_home else event.away_team_abbrev, event.home_team_abbrev if is_home else event.away_team_abbrev)
            opponent = TEAM_ABBR_ALIASES.get(event.away_team_abbrev if is_home else event.home_team_abbrev, event.away_team_abbrev if is_home else event.home_team_abbrev)
            post_margin = (event.home_score - event.away_score) * (1 if is_home else -1)
            pre_margin = post_margin - event.score_value
            assert shot.TEAM_ABBR == team and shot.OPP_ABBR == opponent and bool(shot.HOME_FLAG) == is_home
            assert shot.MARGIN == pre_margin and shot.POINTS == event.score_value
            samples.append({"event_id": shot.EVENT_ID, "game_id": shot.GAME_ID, "season": shot.SEASON_STR, "team": team, "opponent": opponent, "home": bool(is_home), "margin_before_shot": float(pre_margin), "awarded_points": float(event.score_value), "source_text": event.text})
        frames.append(clean)
        summaries.append({"source": source.name, "raw_events": len(raw), "excluded_non_regular_playoff_events": int((~keep).sum()), "shot_rows": len(clean), "unknown_attempt_value_rows": int(clean.SHOT_VALUE.isna().sum()), "quality": quality, "sample_checks": len(sample), "source_quarantine": quarantine})
        print(json.dumps(summaries[-1]), flush=True)
    clean = pd.concat(frames, ignore_index=True)
    clean.to_parquet(output / "shots_clean.parquet", index=False)
    build_and_save_aggregates(clean_path=output / "shots_clean.parquet", output_path=output / "shots_agg.parquet")
    canonical = build_canonical_from_clean(clean)
    (output / "cache").mkdir(exist_ok=True)
    canonical.to_parquet(output / "cache/shots_canonical.parquet", index=False)
    quality = inspect_shot_rows(clean)
    assert quality["valid_attribution"] and quality["missing_margin_rows"] == 0
    now = datetime.now(timezone.utc).isoformat()
    exclusions = {"score_inconsistent_games": sum(q["excluded_games"] for q in quarantines), "score_inconsistent_shots": sum(q["excluded_shots"] for q in quarantines), "location_conflict_rows": int((~clean.LOCATION_VALID).sum()), "geometry_only_zone_rows": int(clean.ZONE_METHOD.eq("geometry_only").sum()), "unknown_attempt_values": int(clean.SHOT_VALUE.isna().sum())}
    artifacts = []
    for path in sorted(output.rglob("*.parquet")):
        version = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {"version": version, "pulled_at": max(a["downloaded_at"] for a in manifest), "source": "ESPN NBA play-by-play via hoopR / sportsdataverse-data", "source_urls": [a["url"] for a in manifest], "raw_source_hashes": {a["name"]: a["sha256"] for a in manifest}, "coverage": "2020-21 through 2025-26; regular season and playoffs, with quarantined games excluded", "generator": "backend/data/etl/rebuild_shots_from_hoopr.py", "created_at": now, "quality_exclusions": exclusions, "notes": "Whole games with inconsistent field-goal score changes are quarantined. Margin is before this shot. Unknown attempt values remain null. Approximate geometry is used only when attempt value is unknown; serious value/location conflicts are excluded from charts and models."}
        path.with_suffix(path.suffix + ".source.json").write_text(json.dumps(metadata, indent=2))
        artifacts.append({"file": str(path.relative_to(output)), "bytes": path.stat().st_size, "version": version})
    report = {"generated_at": now, "seasons": summaries, "quality": quality, "artifacts": artifacts, "spot_checks": samples, "numeric_features": get_feature_spec()["numeric_features"], "excluded_feature": "SHOT_VALUE: older sources leave some missed attempt values unknown", "quality_exclusions": exclusions}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    (output / "source-quarantine.json").write_text(json.dumps({"generated_at": now, "totals": exclusions, "seasons": quarantines}, indent=2))
    print(json.dumps({"quality": quality, "artifacts": artifacts, "spot_checks": len(samples)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rebuild(args.raw_dir, args.output)
