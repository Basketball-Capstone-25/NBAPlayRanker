"""Render any ExportPayload as a simple one-table PDF (SCRUM-485)."""

from __future__ import annotations

import io
from html import escape
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, LongTable, TableStyle


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

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), title=title,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    cell_style = ParagraphStyle("ReportCell", parent=styles["Normal"], fontSize=8, leading=10)
    header_style = ParagraphStyle("ReportHeader", parent=cell_style, textColor=colors.white)
    story = [
        Paragraph(escape(title), styles["Title"]),
        Paragraph(escape(f"Filters: {filters or 'none'}"), styles["Normal"]),
        Spacer(1, 8),
    ]
    # Wrap all content and split wide reports into legible column groups,
    # repeating the first column so rows can be matched between groups.
    columns = list(payload.columns)
    groups = [columns] if len(columns) <= 8 else [
        [columns[0], *columns[start:start + 7]] for start in range(1, len(columns), 7)
    ]
    for group in groups:
        data = [[Paragraph(escape(str(c)), header_style) for c in group]] + [
            [Paragraph(escape(_cell(row.get(c))), cell_style) for c in group]
            for row in payload.rows
        ]
        if not group:
            continue
        table = LongTable(data, repeatRows=1, colWidths=[doc.width / len(group)] * len(group))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([table, Spacer(1, 12)])
    for key, value in payload.metadata.items():
        if key == "title" or value is None:
            continue
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        story.append(Paragraph(escape(f"{key.replace('_', ' ').capitalize()}: {text}"), styles["Normal"]))
        story.append(Spacer(1, 5))
    doc.build(story)
    return buffer.getvalue()
