"""Unified export contract shared by every page that offers a download (SCRUM-484).

One payload shape and one set of rules for CSV, JSON and PDF, so Baseline,
Context/ML and Model Metrics exports stop being separate ad hoc endpoints.
See docs/export-contract.md.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

CONTRACT_VERSION = "1"

MEDIA_TYPES: Dict[str, str] = {
    "csv": "text/csv",
    "json": "application/json",
    "pdf": "application/pdf",
}

PdfRenderer = Callable[["ExportPayload"], bytes]


class ExportFormatUnavailable(ValueError):
    """The requested format is unknown or has no renderer for this report."""


@dataclass(frozen=True)
class ExportPayload:
    """What a page exports: exactly the rows and filters shown on screen."""

    report: str
    filters: Mapping[str, Any]
    columns: Sequence[str]
    rows: Sequence[Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportFile:
    content: bytes
    media_type: str
    filename: str

    @property
    def headers(self) -> Dict[str, str]:
        return {"Content-Disposition": f'attachment; filename="{self.filename}"'}


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value)).strip("-") or "na"


def _normalise_format(format: str) -> str:
    fmt = str(format or "").strip().lower()
    if fmt not in MEDIA_TYPES:
        raise ExportFormatUnavailable(
            f"Unsupported export format '{format}'. Use one of: {', '.join(MEDIA_TYPES)}."
        )
    return fmt


class ExportContract:
    def __init__(self) -> None:
        self._pdf_renderers: Dict[str, PdfRenderer] = {}

    def register_pdf_renderer(self, report: str, renderer: PdfRenderer) -> None:
        self._pdf_renderers[report] = renderer

    def build_export_filename(self, report: str, filters: Mapping[str, Any], format: str) -> str:
        """Deterministic name: report, then filter values in the order given."""
        fmt = _normalise_format(format)
        parts = [_slug(report)] + [_slug(value) for value in filters.values() if value is not None]
        return "_".join(parts) + f".{fmt}"

    def build_export(self, payload: ExportPayload, format: str) -> ExportFile:
        fmt = _normalise_format(format)
        missing = [
            column
            for row in payload.rows
            for column in payload.columns
            if column not in row
        ]
        if missing:
            raise ValueError(f"Rows are missing declared columns: {sorted(set(missing))}")

        if fmt == "csv":
            content = self._to_csv(payload)
        elif fmt == "json":
            content = self._to_json(payload)
        else:
            renderer = self._pdf_renderers.get(payload.report)
            if renderer is None:
                raise ExportFormatUnavailable(f"No PDF renderer registered for '{payload.report}'.")
            content = renderer(payload)

        return ExportFile(
            content=content,
            media_type=MEDIA_TYPES[fmt],
            filename=self.build_export_filename(payload.report, payload.filters, fmt),
        )

    @staticmethod
    def _to_csv(payload: ExportPayload) -> bytes:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(payload.columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload.rows)
        return output.getvalue().encode("utf-8")

    @staticmethod
    def _to_json(payload: ExportPayload) -> bytes:
        rows: List[Dict[str, Any]] = [
            {column: row[column] for column in payload.columns} for row in payload.rows
        ]
        body = {
            "contract_version": CONTRACT_VERSION,
            "report": payload.report,
            "filters": dict(payload.filters),
            "columns": list(payload.columns),
            "rows": rows,
            "metadata": dict(payload.metadata),
        }
        return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
