"""Render any ExportPayload as a simple one-table PDF (SCRUM-485)."""

from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table_pdf(payload: Any) -> bytes:
    """Title, filters and the exported rows, in the payload's column order."""
    styles = getSampleStyleSheet()
    title = str(payload.metadata.get("title") or payload.report.replace("_", " ").title())
    filters = ", ".join(f"{key}={value}" for key, value in payload.filters.items() if value is not None)

    data = [list(payload.columns)] + [
        [_cell(row.get(column)) for column in payload.columns] for row in payload.rows
    ]
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), title=title)
    doc.build([
        Paragraph(title, styles["Title"]),
        Paragraph(f"Filters: {filters or 'none'}", styles["Normal"]),
        Spacer(1, 8),
        table,
    ])
    return buffer.getvalue()
