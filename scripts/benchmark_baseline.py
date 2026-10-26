#!/usr/bin/env python3
"""Bounded coach-baseline timing checks. Tokens are read only from an environment variable."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


THRESHOLDS_MS = {"warm": 5000, "two_concurrent": 10000, "first_after_idle": 45000}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def valid_baseline(payload: object, matchup: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    rows = payload.get("rankings")
    if not isinstance(rows, list) or len(rows) != matchup["k"]:
        return False
    if (payload.get("season"), payload.get("our_team"), payload.get("opp_team")) != (matchup["season"], matchup["our"], matchup["opp"]):
        return False
    plays = []
    scores = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("PLAY_TYPE"), str):
            return False
        score = row.get("PPP_PRED")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            return False
        if not isinstance(row.get("RATIONALE"), str) or not row["RATIONALE"].strip():
            return False
        plays.append(row["PLAY_TYPE"])
        scores.append(score)
    return len(set(plays)) == len(plays) and scores == sorted(scores, reverse=True)


def measure(url: str, token: str, matchup: dict, sample: str) -> dict:
    started_at = utc_now()
    start = time.perf_counter()
    status = None
    valid = False
    error = None
    try:
        request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "Cache-Control": "no-cache"})
        with build_opener(NoRedirect).open(request, timeout=50) as response:
            status = response.status
            body = response.read()
        valid = status == 200 and valid_baseline(json.loads(body), matchup)
        if not valid:
            error = "Unexpected baseline response"
    except HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}"
    except Exception as exc:
        error = type(exc).__name__
    return {
        "sample": sample, "started_at": started_at, "finished_at": utc_now(),
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "status": status, "valid_response": valid, "error": error,
    }


def summarize(samples: list[dict], threshold: int) -> dict:
    elapsed = [sample["elapsed_ms"] for sample in samples]
    observed = p95(elapsed)
    return {
        "requests": len(samples), "p95_ms": observed,
        "min_ms": min(elapsed) if elapsed else None,
        "max_ms": max(elapsed) if elapsed else None,
        "threshold_ms": threshold,
        "pass": bool(samples) and all(sample["valid_response"] for sample in samples) and observed <= threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://nba-playranker-api-937082897804.northamerica-northeast1.run.app")
    parser.add_argument("--token-env", default="NBAPR_BENCHMARK_TOKEN")
    parser.add_argument("--phase", choices=["warm-and-concurrent", "first-after-idle"], default="warm-and-concurrent")
    parser.add_argument("--warm-requests", type=int, default=20)
    parser.add_argument("--concurrent-pairs", type=int, default=5)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--our", default="TOR")
    parser.add_argument("--opp", default="BOS")
    parser.add_argument("--revision", required=True, help="Revision being checked; confirm against request logs afterwards.")
    parser.add_argument("--idle-evidence", type=Path, help="Existing evidence of the quiet interval. Required for first-after-idle.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 5 <= args.warm_requests <= 20 or not 1 <= args.concurrent_pairs <= 5:
        parser.error("Use 5–20 warm requests and 1–5 concurrent pairs.")
    base = urlsplit(args.base_url)
    if base.username or base.password or base.query or base.fragment or base.path not in ["", "/"]:
        parser.error("Use an API origin without credentials, a path or query parameters.")
    allowed_host = "nba-playranker-api-937082897804.northamerica-northeast1.run.app"
    if not ((base.scheme == "https" and base.hostname == allowed_host) or (base.scheme == "http" and base.hostname in ["127.0.0.1", "localhost"])):
        parser.error("Use the known production API or a loopback HTTP server.")
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error(f"Set {args.token_env} to a coach session token without printing it.")
    if args.phase == "first-after-idle" and (not args.idle_evidence or not args.idle_evidence.is_file()):
        parser.error("Provide an existing idle evidence file. A random first request does not prove startup behaviour.")
    matchup = {"season": args.season, "our": args.our, "opp": args.opp, "k": 5, "w_off": 0.7}
    url = args.base_url.rstrip("/") + "/rank-plays/baseline?" + urlencode(matchup)
    report = {
        "endpoint": "/rank-plays/baseline", "base_url": args.base_url,
        "revision_under_test": args.revision, "matchup": matchup,
        "started_at": utc_now(), "timing_scope": "Client elapsed time through complete response body and validation",
        "percentile_method": "Nearest rank; ceil(0.95*n)",
        "thresholds_ms": THRESHOLDS_MS, "samples": [], "results": {},
        "limitations": ["Bounded functional timing sample; not a sustained-load capacity test.", "Confirm the supplied revision and idle/startup state against service request logs."],
    }
    first = measure(url, token, matchup, "first_after_idle" if args.phase == "first-after-idle" else "priming")
    report["samples"].append(first)
    if args.phase == "first-after-idle":
        report["idle_evidence"] = str(args.idle_evidence)
        report["results"]["first_after_idle"] = {
            "elapsed_ms": first["elapsed_ms"], "threshold_ms": THRESHOLDS_MS["first_after_idle"],
            "pass": first["valid_response"] and first["elapsed_ms"] <= THRESHOLDS_MS["first_after_idle"],
        }
    elif first["valid_response"]:
        warm = [measure(url, token, matchup, f"warm_{i + 1}") for i in range(args.warm_requests)]
        concurrent = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for pair in range(args.concurrent_pairs):
                jobs = [pool.submit(measure, url, token, matchup, f"pair_{pair + 1}_{i + 1}") for i in range(2)]
                concurrent.extend(job.result() for job in jobs)
        report["samples"].extend(warm + concurrent)
        report["results"]["warm"] = summarize(warm, THRESHOLDS_MS["warm"])
        report["results"]["two_concurrent"] = summarize(concurrent, THRESHOLDS_MS["two_concurrent"])
        report["startup_state"] = "Not assessed by this phase; priming request is not a cold-start measurement."
    else:
        report["results"]["priming"] = {"pass": False, "reason": "First request failed; remaining load skipped."}
    report["finished_at"] = utc_now()
    report["pass"] = all(result["pass"] for result in report["results"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "pass": report["pass"], "results": report["results"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
