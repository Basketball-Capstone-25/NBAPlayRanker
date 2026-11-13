"""Verify deployed league comparisons against independent raw-source totals."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
TRUSTED_API_ORIGINS = {
    "https://nba-playranker-api-937082897804.northamerica-northeast1.run.app",
    "https://nba-playranker-api-bbv5uwlpba-nn.a.run.app",
    "https://beta-league---nba-playranker-api-bbv5uwlpba-nn.a.run.app",
}
TRUSTED_AUTH = "https://qdodginqubodugfeyiwi.supabase.co"
RATES = ("fg_pct", "efg_pct", "tov_pct", "ft_pct")


def close(actual, expected):
    if expected is None:
        return actual is None
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) \
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def raw_sample(frame):
    if frame.empty:
        return None
    possessions = int(frame.POSS.sum())
    attempts = int(frame.FGA.sum())
    return {
        "possessions": possessions,
        "field_goal_attempts": attempts,
        "fg_pct": float(frame.FGM.sum()) / attempts if attempts else None,
        "efg_pct": float((frame.EFG_PCT * frame.FGA).sum()) / attempts if attempts else None,
        "tov_pct": float((frame.TOV_POSS_PCT * frame.POSS).sum()) / possessions if possessions else None,
        "ft_pct": float((frame.FT_POSS_PCT * frame.POSS).sum()) / possessions if possessions else None,
    }


def expected_rows(raw, season, team, side):
    group = "Offensive" if side == "offense" else "Defensive"
    source = raw.loc[raw.SEASON.eq(season) & raw.SEASON_ID.astype(str).eq("2" + season[:4])
                     & raw.TYPE_GROUPING.eq(group)]
    rows = []
    for play_type in sorted(source.PLAY_TYPE.unique()):
        category = source.loc[source.PLAY_TYPE.eq(play_type)]
        own = raw_sample(category.loc[category.TEAM_ABBREVIATION.eq(team)])
        league = raw_sample(category)
        differences = {rate: 100 * (own[rate] - league[rate])
                       if own and league and own[rate] is not None and league[rate] is not None else None
                       for rate in RATES}
        rows.append({"play_type": play_type, "team": own, "league": league, "difference_pp": differences})
    return rows


def same_rows(actual, expected):
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if actual_row.get("play_type") != expected_row["play_type"]:
            return False
        for side in ("team", "league"):
            observed, wanted = actual_row.get(side), expected_row[side]
            if wanted is None:
                if observed is not None:
                    return False
            elif not isinstance(observed, dict) or not all(close(observed.get(k), v) for k, v in wanted.items()):
                return False
        observed = actual_row.get("difference_pp")
        if not isinstance(observed, dict) or not all(close(observed.get(k), v)
                                                    for k, v in expected_row["difference_pp"].items()):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.api not in TRUSTED_API_ORIGINS:
        raise ValueError("Use a listed NBA Play Ranker Cloud Run origin.")
    config = dotenv_values(REPO / ".env.local")
    auth = config["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    if auth != TRUSTED_AUTH:
        raise ValueError("Use the authorized capstone authentication project.")
    key = config["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
    accounts = json.loads((Path.home() / ".config/nbaplayranker/test-accounts-20261003.json").read_text())
    source = REPO / "backend/data/synergy_playtypes_2019_2025_players.csv"
    raw = pd.read_csv(source)
    sessions, checks = {}, []
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "api": args.api,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "checks": checks, "scenarios": []}
    args.output.mkdir(parents=True, exist_ok=False)

    def check(name, ok, **details):
        checks.append({"name": name, "passed": bool(ok), **details})
        if not ok:
            raise AssertionError(name)

    def call(name, path, role=None, params=None, status=200):
        response = requests.get(args.api + path, params=params, headers=sessions.get(role, {}),
                                timeout=60, allow_redirects=False)
        check(name, response.status_code == status, status=response.status_code)
        return response

    try:
        check("service ready", call("readiness response", "/ready").json().get("ready") is True)
        for role in ("coach", "analyst"):
            login = requests.post(auth + "/auth/v1/token?grant_type=password", headers={"apikey": key},
                                  json={field: accounts[role][field] for field in ("email", "password")},
                                  timeout=30, allow_redirects=False)
            check(role + " sign in", login.status_code == 200)
            sessions[role] = {"Authorization": "Bearer " + login.json()["access_token"]}
        sessions["invalid"] = {"Authorization": "Bearer invalid"}
        path = "/analytics/playtypes/league-comparison"
        base = {"season": "2024-25", "team": "TOR", "side": "offense"}
        call("anonymous comparison denied", path, params=base, status=401)
        call("invalid token denied", path, "invalid", base, 401)
        call("analyst comparison denied", path, "analyst", base, 403)
        for season in ("2024-25", "2023-24"):
            for team in ("TOR", "BOS"):
                for side in ("offense", "defense"):
                    filters = {"season": season, "team": team, "side": side}
                    label = f"{season}-{team}-{side}"
                    response = call(label + " response", path, "coach", filters)
                    data = response.json()
                    check(label + " current scope", data.get("status") == "ok"
                          and data.get("filters") == {**filters, "season_phase": "regular_season"}
                          and data.get("selected_team_in_league") is True)
                    check(label + " source and private headers",
                          data.get("dataset_version") == "sha256:" + report["source_sha256"]
                          and data.get("aggregation_version") == "synergy_regular_season_v2"
                          and response.headers.get("Cache-Control") == "private, no-store"
                          and response.headers.get("X-Data-Quality") == "pass")
                    expected = expected_rows(raw, **filters)
                    check(label + " raw counts rates and differences", same_rows(data.get("rows"), expected),
                          play_types=len(expected), samples=2 * len(expected), compared_rates=8 * len(expected))
                    report["scenarios"].append({"filters": filters, "play_type_count": len(expected),
                        "expected_table_body_rows": 3 * len(expected),
                        "team_possessions": sum(row["team"]["possessions"] for row in expected if row["team"]),
                        "team_field_goal_attempts": sum(row["team"]["field_goal_attempts"] for row in expected if row["team"]),
                        "first_row": expected[0]})
                    (args.output / (label + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        empty = call("unknown season response", path, "coach", {**base, "season": "2028-29"}).json()
        check("unknown season stays unavailable", empty.get("status") == "no_data" and empty.get("rows") == []
              and empty.get("filters") == {**base, "season": "2028-29", "season_phase": "regular_season"})
        for change, status in [({"season": "2024-99"}, 400), ({"team": "XXX"}, 400),
                               ({"side": "both"}, 422), ({"team": ""}, 422), ({"season": "bad"}, 422)]:
            call("invalid filter " + str(change), path, "coach", {**base, **change}, status)
        summary_filters = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 5, "w_off": 0.7}
        summary = call("coach summary smoke", "/analytics/playtypes/summary", "coach", summary_filters).json()
        analyst_summary = call("analyst summary smoke", "/analytics/playtypes/summary", "analyst", summary_filters).json()
        check("summary roles still agree", summary == analyst_summary and summary.get("status") == "ok")
        team_source = raw.loc[raw.SEASON_ID.astype(str).eq("22024") & raw.TYPE_GROUPING.eq("Offensive")
                              & raw.TEAM_ABBREVIATION.eq("TOR")]
        counts, values = int(team_source.POSS.sum()), summary["summary"]
        check("summary source and arithmetic remain intact",
              summary.get("dataset_version") == "sha256:" + report["source_sha256"]
              and close(values["team_possessions"], counts)
              and close(values["observed_team_ppp"], float(team_source.PTS.sum()) / counts)
              and close(values["modeled_uplift_ppp"], values["modeled_topk_ppp"] - values["observed_team_ppp"])
              and close(values["selected_usage_pct"], sum(row["usage_pct"] for row in summary["play_types"])))
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        if isinstance(error, AssertionError):
            report["failed_check"] = str(error)
    finally:
        report["session_cleanup"] = []
        for role in ("coach", "analyst"):
            if role not in sessions:
                continue
            try:
                response = requests.post(auth + "/auth/v1/logout?scope=local",
                    headers={"apikey": key, **sessions[role]}, timeout=30, allow_redirects=False)
                status = response.status_code
            except requests.RequestException:
                status = "unconfirmed"
            report["session_cleanup"].append({"role": role, "status": status})
            checks.append({"name": role + " session cleaned up", "passed": status == 204, "status": status})
        report["passed"] = report.get("complete", False) and all(item["passed"] for item in checks)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": len(checks),
                          "failed_check": report.get("failed_check"), "error_type": report.get("error_type")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
