"""Verify the Alpha API with real sessions and independent source totals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from dotenv import dotenv_values
import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlparse(args.api)
    if parsed.scheme != "https" or not parsed.hostname.endswith(".run.app") or parsed.path or parsed.query:
        raise ValueError("Use an HTTPS Cloud Run service origin.")
    repo = Path(__file__).resolve().parents[1]
    config = dotenv_values(repo / ".env.local")
    auth = config["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    key = config["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
    accounts = json.loads((Path.home() / ".config/nbaplayranker/test-accounts-20261003.json").read_text())
    source = repo / "backend/data/synergy_playtypes_2019_2025_players.csv"
    raw = pd.read_csv(source)
    sessions, checks = {}, []
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "api": args.api,
              "revision": args.revision, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "checks": checks, "session_cleanup": []}
    args.output.mkdir(parents=True, exist_ok=True)

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})
        if not passed:
            raise AssertionError(name)

    def call(name, route, role=None, params=None, expected=200):
        start = time.perf_counter()
        response = requests.get(args.api + route, params=params, headers=sessions.get(role, {}), timeout=60)
        check(name, response.status_code == expected, status=response.status_code,
              elapsed_ms=round((time.perf_counter() - start) * 1000, 2))
        return response

    def same(name, actual, expected):
        check(name, math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10))

    try:
        check("service ready", call("readiness", "/ready").json().get("ready") is True)
        for role in ("coach", "analyst"):
            result = requests.post(auth + "/auth/v1/token?grant_type=password", headers={"apikey": key},
                                   json={k: accounts[role][k] for k in ("email", "password")}, timeout=30)
            check(role + " sign-in", result.status_code == 200)
            sessions[role] = {"Authorization": "Bearer " + result.json()["access_token"]}
        filters = {"season": "2024-25", "our": "TOR", "opp": "BOS", "min_poss": 20}
        call("anonymous dashboard denied", "/analytics/playtypes", params=filters, expected=401)
        call("analyst dashboard denied", "/analytics/playtypes", "analyst", filters, 403)
        sessions["invalid"] = {"Authorization": "Bearer invalid-session"}
        call("invalid session denied", "/analytics/playtypes", "invalid", filters, 401)
        for season, our, opp in (("2024-25", "TOR", "BOS"), ("2019-20", "LAL", "SAS")):
            selected = {**filters, "season": season, "our": our, "opp": opp}
            response = call(f"dashboard {season} {our}/{opp}", "/analytics/playtypes", "coach", selected)
            body = response.json()
            check("scope and version " + season, body["filters"]["season_phase"] == "regular_season"
                  and body["aggregation_version"] == "synergy_regular_season_v2"
                  and body["dataset_versions"]["synergy"] == "sha256:" + report["source_sha256"])
            check("private response " + season, response.headers.get("Cache-Control") == "private, no-store")
            regular = raw.loc[raw.SEASON.eq(season) & raw.SEASON_ID.astype(str).eq("2" + season[:4])]
            offense = regular.loc[regular.TYPE_GROUPING.eq("Offensive")]
            defense = regular.loc[regular.TYPE_GROUPING.eq("Defensive")]
            team = offense.loc[offense.TEAM_ABBREVIATION.eq(our)]
            opponent = defense.loc[defense.TEAM_ABBREVIATION.eq(opp)]
            same("offensive total " + season, body["usage_efficiency"]["total_possessions"], team.POSS.sum())
            same("defensive total " + season, body["opponent_defense"]["tracked_possessions"], opponent.POSS.sum())
            for row in body["usage_efficiency"]["rows"]:
                play = row["play_type"]
                t = team.loc[team.PLAY_TYPE.eq(play)]
                league = offense.loc[offense.PLAY_TYPE.eq(play)]
                for metric, expected in {"possessions": t.POSS.sum(), "points": t.PTS.sum(),
                                         "ppp": t.PTS.sum()/t.POSS.sum(), "usage_pct": 100*t.POSS.sum()/team.POSS.sum(),
                                         "league_ppp": league.PTS.sum()/league.POSS.sum(),
                                         "league_usage_pct": 100*league.POSS.sum()/offense.POSS.sum()}.items():
                    same(f"{season} {our} {play} {metric}", row[metric], expected)
            rows = body["opponent_defense"]["rows"]
            for row in rows:
                play = row["play_type"]
                t = opponent.loc[opponent.PLAY_TYPE.eq(play)]
                league = defense.loc[defense.PLAY_TYPE.eq(play)]
                for metric, expected in {"possessions": t.POSS.sum(), "points_allowed": t.PTS.sum(),
                                         "allowed_ppp": t.PTS.sum()/t.POSS.sum(),
                                         "league_allowed_ppp": league.PTS.sum()/league.POSS.sum()}.items():
                    same(f"{season} {opp} {play} {metric}", row[metric], expected)
            check("sorted defense " + season, rows == sorted(rows, key=lambda r: -r["allowed_ppp_delta"]))
            check("missing defensive categories " + season,
                  body["coverage"]["unavailable_defense_types"] == ["Cut", "Misc", "OffRebound", "Transition"])
            (args.output / f"dashboard-{season}-{our}-{opp}.json").write_text(json.dumps(body, indent=2) + "\n")
        high = call("high sample cutoff", "/analytics/playtypes", "coach", {**filters, "min_poss": 10000}).json()
        check("all rows retained with no highlights", len(high["usage_efficiency"]["rows"]) == 11
              and len(high["opponent_defense"]["rows"]) == 7
              and all(not r["sample_ok"] for s in ("usage_efficiency", "opponent_defense") for r in high[s]["rows"])
              and all(r["weak_spot_rank"] is None for r in high["opponent_defense"]["rows"]))
        for changed, expected in (({"min_poss": 0}, 422), ({"opp": "TOR"}, 400), ({"season": "1900-01"}, 400)):
            call("invalid input " + str(changed), "/analytics/playtypes", "coach", {**filters, **changed}, expected)
        rank_params = {k: v for k, v in filters.items() if k != "min_poss"}
        rank_params["k"] = 5
        for route, role, params in (("/rank-plays/baseline", "coach", rank_params),
                                    ("/rank-plays/context-ml", "coach", {**rank_params, "margin": -3, "period": 4, "time_remaining": 60}),
                                    ("/data/team-playtypes", "analyst", {"season": "2024-25", "team": "TOR", "side": "offense"}),
                                    ("/metrics/topk-uplift", "analyst", rank_params),
                                    ("/metrics/baseline-vs-ml", "analyst", {}), ("/analysis/ml", "analyst", {})):
            call("existing route " + route, route, role, params)
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        if isinstance(error, AssertionError): report["failed_check"] = str(error)
    finally:
        for role in ("coach", "analyst"):
            if role in sessions:
                try:
                    result = requests.post(auth + "/auth/v1/logout?scope=local", headers={"apikey": key, **sessions[role]}, timeout=30)
                    report["session_cleanup"].append({"role": role, "status": result.status_code})
                except requests.RequestException:
                    report["session_cleanup"].append({"role": role, "status": "unconfirmed"})
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = report.get("complete", False) and all(c["passed"] for c in checks)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed_check": report.get("failed_check")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
