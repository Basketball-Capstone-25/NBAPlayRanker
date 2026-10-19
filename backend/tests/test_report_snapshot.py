"""Displayed report downloads preserve values and enforce access and size limits."""

from __future__ import annotations

import csv
import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from application.api_coordination import app as app_module
from application.api_coordination.report_snapshot import MAX_SNAPSHOT_BYTES

AUTH = "application.access_control_services.access_control_service"


def _body(report="context_rankings", fmt="json"):
    return {
        "format": fmt,
        "payload": {
            "report": report,
            "filters": {"season": "2024-25", "our": "TOR", "opp": "BOS", "quarter": 4},
            "columns": ["Play type", "Final PPP", "Overlay PPP"],
            "rows": [{"Play type": "Transition", "Final PPP": 1.2345, "Overlay PPP": -0.02}],
            "metadata": {"title": "Current results", "limitations": ["Context adjustment is a scenario estimate."]},
        },
    }


def _post(body=None, role="coach", headers=None, content=None):
    with patch(f"{AUTH}.decode_supabase_jwt", return_value={"sub": "test-user"}), patch(
        f"{AUTH}.get_profile_role", return_value=role,
    ):
        return TestClient(app_module.app).post(
            "/export/report", json=body, content=content,
            headers=headers if headers is not None else {"Authorization": "Bearer test"},
        )


@pytest.mark.parametrize("report,role", [
    ("baseline_rankings", "coach"), ("context_rankings", "coach"), ("model_metrics", "analyst"),
])
@pytest.mark.parametrize("fmt,media", [("csv", "text/csv"), ("json", "application/json"), ("pdf", "application/pdf")])
def test_report_formats_keep_the_displayed_results(report, role, fmt, media):
    body = _body(report, fmt)
    res = _post(body, role)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith(media)
    assert res.headers["content-disposition"].endswith(f'.{fmt}"')
    assert res.headers["cache-control"] == "private, no-store"
    if fmt == "json":
        assert res.json()["rows"] == body["payload"]["rows"]
        assert res.json()["filters"] == body["payload"]["filters"]
        assert res.json()["metadata"]["export_source"] == "displayed_result_snapshot"
    elif fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(res.text)))
        assert rows == [{"Play type": "Transition", "Final PPP": "1.2345", "Overlay PPP": "-0.02"}]
    else:
        assert res.content.startswith(b"%PDF") and res.content.rstrip().endswith(b"%%EOF")


@pytest.mark.parametrize("report,role", [
    ("baseline_rankings", "analyst"), ("context_rankings", "analyst"),
    ("model_metrics", "coach"), ("model_metrics", "pending"),
])
def test_role_mismatch_never_returns_an_attachment(report, role):
    res = _post(_body(report), role)
    assert res.status_code == 403
    assert "content-disposition" not in res.headers


def test_missing_session_is_rejected():
    res = _post(_body(), headers={})
    assert res.status_code == 401
    assert "content-disposition" not in res.headers


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(format="xlsx"),
    lambda b: b["payload"].update(report="arbitrary_report"),
    lambda b: b["payload"].update(rows=[]),
    lambda b: b["payload"].update(rows=[{}]),
    lambda b: b["payload"].update(rows=[b["payload"]["rows"][0]] * 201),
    lambda b: b["payload"].update(columns=["x"] * 17),
    lambda b: b["payload"].update(columns=["Play type", "Play type"]),
    lambda b: b["payload"]["rows"][0].update(extra="hidden value"),
    lambda b: b["payload"]["rows"][0].update({"Play type": "x" * 257}),
    lambda b: b["payload"]["rows"][0].update({"Final PPP": {"nested": 1}}),
    lambda b: b["payload"].update(filters={"season": [1, 2]}),
    lambda b: b["payload"].update(metadata={"details": {"a": {"b": {"c": {"d": [1]}}}}}),
])
def test_invalid_snapshot_is_rejected_without_a_partial_file(mutation):
    body = _body()
    mutation(body)
    res = _post(body)
    assert res.status_code == 400, res.text
    assert "content-disposition" not in res.headers


def test_body_limit_is_enforced_while_reading():
    res = _post(content=b" " * (MAX_SNAPSHOT_BYTES + 1), headers={
        "Authorization": "Bearer test", "Content-Type": "application/json",
    })
    assert res.status_code == 413
    assert "content-disposition" not in res.headers


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_values_are_rejected(number):
    raw = json.dumps(_body()).replace("1.2345", number)
    res = _post(content=raw, headers={"Authorization": "Bearer test", "Content-Type": "application/json"})
    assert res.status_code == 400


def test_non_json_requests_are_rejected():
    assert _post(content="not json").status_code == 415


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_escaped_lone_surrogates_are_rejected_before_rendering(fmt):
    body = _body(fmt=fmt)
    body["payload"]["rows"][0]["Play type"] = "\ud800"
    res = _post(content=json.dumps(body).encode("ascii"), headers={
        "Authorization": "Bearer test", "Content-Type": "application/json",
    })
    assert res.status_code == 400
    assert "content-disposition" not in res.headers


def test_csv_formula_strings_are_escaped_but_numbers_keep_their_values():
    body = _body(fmt="csv")
    body["payload"]["rows"][0]["Play type"] = '=HYPERLINK("https://example.org")'
    rows = list(csv.DictReader(io.StringIO(_post(body).text)))
    assert rows[0]["Play type"].startswith("'=")
    assert rows[0]["Overlay PPP"] == "-0.02"


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_context_filenames_distinguish_trailing_and_leading_margins(fmt):
    body = _body(fmt=fmt)
    body["payload"]["filters"]["margin"] = -2
    behind = _post(body)
    body["payload"]["filters"]["margin"] = 2
    ahead = _post(body)

    assert behind.status_code == ahead.status_code == 200
    assert behind.headers["content-disposition"] == (
        f'attachment; filename="context-rankings_2024-25_TOR_BOS_4_-2.{fmt}"'
    )
    assert ahead.headers["content-disposition"] == (
        f'attachment; filename="context-rankings_2024-25_TOR_BOS_4_2.{fmt}"'
    )
    if fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(behind.text)))
        assert rows[0]["Overlay PPP"] == "-0.02"
    elif fmt == "json":
        assert behind.json()["filters"]["margin"] == -2
        assert ahead.json()["filters"]["margin"] == 2
    else:
        assert behind.content.startswith(b"%PDF") and ahead.content.startswith(b"%PDF")


def test_pdf_handles_wide_columns_and_literal_markup():
    body = _body(fmt="pdf")
    body["payload"]["metadata"]["title"] = "Literal <tag> & report"
    body["payload"]["columns"] = [f"Long column {n}" for n in range(16)]
    body["payload"]["rows"] = [{c: "<img src='https://example.org'>" for c in body["payload"]["columns"]}]
    res = _post(body)
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")


def test_long_filters_do_not_create_oversized_attachment_headers():
    body = _body()
    body["payload"]["filters"] = {f"filter{n}": "x" * 256 for n in range(24)}
    res = _post(body)
    assert res.status_code == 200
    assert len(res.headers["content-disposition"]) < 220


def test_client_cannot_override_snapshot_provenance():
    body = _body()
    body["payload"]["metadata"]["export_source"] = "verified server forecast"
    assert _post(body).json()["metadata"]["export_source"] == "displayed_result_snapshot"
