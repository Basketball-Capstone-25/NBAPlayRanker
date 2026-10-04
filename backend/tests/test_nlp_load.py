"""Multi-user load and memory check for the NLP endpoints (SCRUM-502).

Simulates concurrent coaches calling /nlp/parse and /nlp/explain against the
full FastAPI app in one process, and records process memory so the result can
be compared with the 2 GiB Cloud Run limit. Results are written to
docs/nlp-load-test-results.json.
"""

from __future__ import annotations

import json
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

psutil = pytest.importorskip("psutil")

from application.api_coordination.app import app  # noqa: E402
from infrastructure.external_integrations.nlp_pipeline import pipeline_health  # noqa: E402

CONCURRENT_USERS = 8
REQUESTS_PER_USER = 50
MEMORY_LIMIT_MB = 2048  # Cloud Run instance memory
RESULTS_PATH = Path(__file__).resolve().parents[2] / "docs" / "nlp-load-test-results.json"

PROMPTS = [
    "Down 3 with 0:28 left in Q4, need a quick 2, they're switching everything",
    "Tie game, 1:45 in the 3rd, get a stop",
    "Up by 5, 2 minutes left, burn clock, vs 2-3 zone",
    "ATO, down 3, 18 on the shot clock, 0:32 in OT, need a clean 3",
    "Just win the game",
    "leading by 3 late in the 4th after timeout",
]

EXPLAIN_PAYLOAD = {
    "context": {"period": 4, "time_remaining": 28, "margin": -3, "need": "quick2"},
    "ranked_context": {
        "rankings": [
            {"PLAY_TYPE": "P&R Ball Handler", "PPP_CONTEXT": 1.08, "PPP_BASELINE": 1.00},
            {"PLAY_TYPE": "Post Up", "PPP_CONTEXT": 1.02, "PPP_BASELINE": 1.00},
        ]
    },
}


def _rss_mb(process) -> float:
    return process.memory_info().rss / (1024 * 1024)


def _user_session(user: int) -> list[tuple[int, float]]:
    """One simulated coach: alternate parse and explain calls."""
    client = TestClient(app)
    samples = []
    for i in range(REQUESTS_PER_USER):
        started = time.perf_counter()
        if i % 5 == 4:
            response = client.post("/nlp/explain", json=EXPLAIN_PAYLOAD)
        else:
            response = client.post("/nlp/parse", json={"text": PROMPTS[(user + i) % len(PROMPTS)]})
        samples.append((response.status_code, time.perf_counter() - started))
    return samples


@pytest.mark.integration
def test_nlp_endpoints_survive_concurrent_users_within_memory_limit():
    process = psutil.Process()
    rss_before_nlp = _rss_mb(process)

    warmup = TestClient(app).post("/nlp/parse", json={"text": PROMPTS[0]})
    assert warmup.status_code == 200
    rss_after_warmup = _rss_mb(process)

    peak = {"rss": rss_after_warmup}
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], _rss_mb(process))
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as pool:
        sessions = list(pool.map(_user_session, range(CONCURRENT_USERS)))
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=2)

    samples = [sample for session in sessions for sample in session]
    latencies_ms = sorted(seconds * 1000 for _, seconds in samples)
    errors = [status for status, _ in samples if status != 200]
    rss_after_load = _rss_mb(process)
    peak_rss = max(peak["rss"], rss_after_load)

    results = {
        "scenario": "in-process FastAPI app, threads as concurrent users",
        "concurrent_users": CONCURRENT_USERS,
        "requests_total": len(samples),
        "errors": len(errors),
        "duration_seconds": round(elapsed, 2),
        "requests_per_second": round(len(samples) / elapsed, 1),
        "latency_ms": {
            "p50": round(statistics.median(latencies_ms), 1),
            "p95": round(latencies_ms[int(len(latencies_ms) * 0.95) - 1], 1),
            "max": round(latencies_ms[-1], 1),
        },
        "memory_mb": {
            "rss_before_first_nlp_request": round(rss_before_nlp, 1),
            "rss_after_nlp_warmup": round(rss_after_warmup, 1),
            "rss_peak_under_load": round(peak_rss, 1),
            "rss_after_load": round(rss_after_load, 1),
            "limit": MEMORY_LIMIT_MB,
        },
        "nlp_pipeline": pipeline_health(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    assert not errors, f"{len(errors)} failed requests: {sorted(set(errors))}"
    assert peak_rss < MEMORY_LIMIT_MB, f"peak RSS {peak_rss:.0f} MB exceeds the {MEMORY_LIMIT_MB} MB limit"
    assert rss_after_load - rss_after_warmup < 200, "memory grew by more than 200 MB under load"
    assert results["latency_ms"]["p95"] < 2000
