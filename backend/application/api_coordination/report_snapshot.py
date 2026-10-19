"""Validate the displayed results submitted by the three report pages."""

from __future__ import annotations

import json
import math
from typing import Any

from application.api_coordination.export_contract import ExportPayload

MAX_SNAPSHOT_BYTES = 256 * 1024
SNAPSHOT_ROLES = {
    "baseline_rankings": "recommendation",
    "context_rankings": "recommendation",
    "model_metrics": "analytics",
}


def _scalar(value: Any, limit: int = 256) -> bool:
    return (
        value is None
        or isinstance(value, bool)
        or isinstance(value, str) and len(value) <= limit
        or type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e100
    )


def _metadata(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if _scalar(value, 2000):
        return True
    if isinstance(value, list):
        return len(value) <= 64 and all(_metadata(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return len(value) <= 64 and all(
            isinstance(k, str) and len(k) <= 64 and _metadata(v, depth + 1)
            for k, v in value.items()
        )
    return False


def validate_snapshot(body: Any) -> tuple[ExportPayload, str]:
    """A snapshot is an export of displayed values, not a new model evaluation."""
    # JSON permits escaped lone surrogates, but none of the download encodings
    # can represent them. Reject them here rather than failing during rendering.
    json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not isinstance(body, dict) or set(body) != {"format", "payload"}:
        raise ValueError("Send a format and report payload.")
    if body["format"] not in ("csv", "json", "pdf"):
        raise ValueError("Choose CSV, JSON or PDF.")
    data = body["payload"]
    required = {"report", "filters", "columns", "rows"}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"metadata"}:
        raise ValueError("The report payload is incomplete.")
    if not isinstance(data["report"], str) or data["report"] not in SNAPSHOT_ROLES:
        raise ValueError("This report cannot be exported.")
    filters = data["filters"]
    if not isinstance(filters, dict) or len(filters) > 24 or not all(
        isinstance(k, str) and 0 < len(k) <= 64 and _scalar(v) for k, v in filters.items()
    ):
        raise ValueError("Report filters must be short text or numbers.")
    columns = data["columns"]
    if not isinstance(columns, list) or not 1 <= len(columns) <= 16 or not all(
        isinstance(c, str) and 0 < len(c) <= 64 for c in columns
    ) or len(set(columns)) != len(columns):
        raise ValueError("A report needs 1 to 16 distinct columns.")
    rows = data["rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 200 or not all(
        isinstance(row, dict) and set(row) == set(columns)
        and all(_scalar(value) for value in row.values()) for row in rows
    ):
        raise ValueError("A report needs 1 to 200 complete rows with short text or numbers.")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or not _metadata(metadata) or len(
        json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    ) > 8192:
        raise ValueError("Report details are too large or contain unsupported values.")
    # Set this on the server: a browser snapshot must never masquerade as a
    # fresh server calculation or an independently verified forecast result.
    metadata = {**metadata, "export_source": "displayed_result_snapshot"}
    return ExportPayload(data["report"], filters, columns, rows, metadata), body["format"]
