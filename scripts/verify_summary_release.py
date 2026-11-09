"""Check the deployed summary against source totals with both test accounts."""
import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

TRUSTED_API_ORIGINS = {
    "https://nba-playranker-api-937082897804.northamerica-northeast1.run.app",
    "https://nba-playranker-api-bbv5uwlpba-nn.a.run.app",
    "https://beta-summary---nba-playranker-api-bbv5uwlpba-nn.a.run.app",
}


def clutch_csv_rows(text):
    # Metadata can contain commas and is quoted by csv.writer.
    records = [row for row in csv.reader(io.StringIO(text)) if row]
    metadata = [row[0] for row in records if row[0].startswith("#")]
    table = [row for row in records if not row[0].startswith("#")]
    return metadata, table[0] if table else [], table[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.api not in TRUSTED_API_ORIGINS:
        raise ValueError("Use a listed NBA Play Ranker Cloud Run origin.")
    repo = Path(__file__).resolve().parents[1]
    config = dotenv_values(repo / ".env.local")
    auth = config["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    key = config["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
    accounts = json.loads((Path.home() / ".config/nbaplayranker/test-accounts-20261003.json").read_text())
    source = repo / "backend/data/synergy_playtypes_2019_2025_players.csv"
    raw = pd.read_csv(source)
    sessions, checks = {}, []
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "api": args.api,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "checks": checks}
    args.output.mkdir(parents=True, exist_ok=False)

    def check(name, ok, **details):
        checks.append({"name": name, "passed": bool(ok), **details})
        if not ok:
            raise AssertionError(name)

    def call(name, path, role=None, params=None, expected=200):
        response = requests.get(args.api + path, params=params, headers=sessions.get(role, {}), timeout=60,
                                allow_redirects=False)
        check(name, response.status_code == expected, status=response.status_code)
        return response

    def same(name, actual, expected):
        check(name, math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

    try:
        check("ready", call("readiness", "/ready").json()["ready"])
        for role in ("coach", "analyst"):
            login = requests.post(auth + "/auth/v1/token?grant_type=password", headers={"apikey": key},
                                  json={k: accounts[role][k] for k in ("email", "password")}, timeout=30,
                                  allow_redirects=False)
            check(role + " sign in", login.status_code == 200)
            sessions[role] = {"Authorization": "Bearer " + login.json()["access_token"]}
        path = "/analytics/playtypes/summary"
        base = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 5, "w_off": 0.7}
        call("anonymous summary denied", path, params=base, expected=401)
        sessions["invalid"] = {"Authorization": "Bearer invalid"}
        call("invalid token denied", path, "invalid", base, 401)
        for season, opponent, k in [("2024-25", "BOS", 5), ("2023-24", "BOS", 3), ("2024-25", "DEN", 5)]:
            filters = {**base, "season": season, "opp": opponent, "k": k}
            label = f"{season}-{opponent}-k{k}"
            coach = call(label + " coach", path, "coach", filters)
            data = coach.json()
            check(label + " role parity", data == call(label + " analyst", path, "analyst", filters).json())
            check(label + " private", coach.headers.get("Cache-Control") == "private, no-store")
            check(label + " scope", data["status"] == "ok" and data["filters"]["period"] == "season"
                  and data["filters"]["season_phase"] == "regular_season")
            check(label + " source", data["dataset_version"] == "sha256:" + report["source_sha256"])
            totals = raw.loc[raw.SEASON.eq(season) & raw.SEASON_ID.astype(str).eq("2" + season[:4])
                             & raw.TEAM_ABBREVIATION.eq("TOR") & raw.TYPE_GROUPING.eq("Offensive")]
            summary = data["summary"]
            same(label + " possession denominator", summary["team_possessions"], totals.POSS.sum())
            same(label + " observed PPP", summary["observed_team_ppp"], totals.PTS.sum() / totals.POSS.sum())
            baseline = call(label + " rankings", "/rank-plays/baseline", "coach", filters).json()["rankings"]
            selected = {row["PLAY_TYPE"]: row for row in baseline}
            check(label + " selected types", list(selected) == [r["play_type"] for r in data["play_types"]])
            weighted = sum(r["PPP_PRED"] * r["POSS_OFF"] for r in baseline) / sum(r["POSS_OFF"] for r in baseline)
            same(label + " modeled PPP", summary["modeled_topk_ppp"], weighted)
            same(label + " difference", summary["modeled_uplift_ppp"], weighted - summary["observed_team_ppp"])
            for row in data["play_types"]:
                poss = totals.loc[totals.PLAY_TYPE.eq(row["play_type"]), "POSS"].sum()
                same(label + " usage " + row["play_type"], row["usage_pct"], 100 * poss / totals.POSS.sum())
            same(label + " selected usage", summary["selected_usage_pct"], sum(r["usage_pct"] for r in data["play_types"]))
            (args.output / (label + ".json")).write_text(json.dumps(data, indent=2) + "\n")
        empty = call("missing period", path, "analyst", {**base, "season": "2028-29"}).json()
        check("missing data stays empty", empty["status"] == "no_data" and empty["summary"] is None and empty["play_types"] == [])
        for change, status in [({"opp": "TOR"}, 400), ({"period": "last10"}, 422), ({"k": 11}, 422)]:
            call("invalid selection " + str(change), path, "coach", {**base, **change}, status)
        call("detailed evidence stays analyst only", "/metrics/topk-uplift", "coach", base, 403)
        clutch = {"season": "2024-25", "team": "TOR", "opp": "BOS", "period_from": 4,
                  "seconds_left": 300, "margin_min": -5, "margin_max": 5}
        body = call("clutch report", "/data/clutch-report", "analyst", clutch).json()
        for extension in ("csv", "pdf"):
            response = call("clutch " + extension, "/data/clutch-report." + extension, "analyst", clutch)
            (args.output / ("clutch-direct." + extension)).write_bytes(response.content)
            check(extension + " attachment", response.headers.get("Content-Disposition") ==
                  f'attachment; filename="clutch_TOR_vs_BOS_2024-25.{extension}"')
            check(extension + " private", response.headers.get("Cache-Control") == "private, no-store")
            if extension == "pdf":
                check("PDF media type", response.headers.get("Content-Type", "").startswith("application/pdf"))
                check("PDF signature", response.content.startswith(b"%PDF-") and b"%%EOF" in response.content[-1024:])
            else:
                check("CSV media type", response.headers.get("Content-Type", "").startswith("text/csv"))
                metadata, header, rows = clutch_csv_rows(response.text)
                check("CSV metadata", len(metadata) == 2 and metadata[0].startswith("# Clutch report TOR vs BOS 2024-25:")
                      and str(body["small_sample_threshold"]) in metadata[1])
                check("CSV columns", header == ["zone", "team", "shots", "makes", "points", "points_per_shot", "small_sample"])
                check("CSV rows", len(rows) == 2 * len(body["rows"]))
                expected_rows = []
                for row in body["rows"]:
                    for side, team in (("team", clutch["team"]), ("opp", clutch["opp"])):
                        split = row[side]
                        pps = "" if split["points_per_shot"] is None else f'{split["points_per_shot"]:.3f}'
                        expected_rows.append([row["label"], team, str(split["shots"]), str(split["makes"]),
                                              str(split["points"]), pps, "yes" if split["small_sample"] else "no"])
                check("CSV values match report", rows == expected_rows, data_rows=len(rows))
        call("coach clutch denied", "/data/clutch-report", "coach", clutch, 403)
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        if isinstance(error, AssertionError): report["failed_check"] = str(error)
    finally:
        report["session_cleanup"] = []
        for role in ("coach", "analyst"):
            if role in sessions:
                try:
                    response = requests.post(auth + "/auth/v1/logout?scope=local", headers={"apikey": key, **sessions[role]},
                                             timeout=30, allow_redirects=False)
                    report["session_cleanup"].append({"role": role, "status": response.status_code})
                except requests.RequestException:
                    report["session_cleanup"].append({"role": role, "status": "unconfirmed"})
        for cleanup in report["session_cleanup"]:
            checks.append({"name": cleanup["role"] + " session cleaned up", "passed": cleanup["status"] == 204,
                           "status": cleanup["status"]})
        report["passed"] = report.get("complete", False) and all(c["passed"] for c in checks)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed_check": report.get("failed_check")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
