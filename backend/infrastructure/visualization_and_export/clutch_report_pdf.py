"""SCRUM-629: the clutch report as a one-page PDF, both teams side by side by zone."""

from __future__ import annotations

import io
from html import escape
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SMALL_FILL = colors.HexColor("#fff4d6")


def _pps(split: Dict[str, Any]) -> str:
    value = split["points_per_shot"]
    return "-" if value is None else f"{value:.2f}"


def _shots(split: Dict[str, Any]) -> str:
    return f"{split['shots']}*" if split["small_sample"] else str(split["shots"])


def render_clutch_pdf(report: Dict[str, Any], situation: str) -> bytes:
    f = report["filters"]
    styles = getSampleStyleSheet()
    title = f"Clutch report: {f['team']} vs {f['opp']}, {f['season']}"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=title,
                            leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    story = [
        Paragraph(escape(title), styles["Title"]),
        Paragraph(escape(f"Situations: {situation}."), styles["Normal"]),
        Spacer(1, 10),
    ]
    if not report["rows"]:
        story.append(Paragraph(escape(report.get("message") or "No shots match these situations."), styles["Normal"]))
    else:
        data = [
            ["", f["team"], "", "", f["opp"], "", ""],
            ["Zone", "Shots", "Makes", "Pts/shot", "Shots", "Makes", "Pts/shot"],
        ]
        small = []
        for i, row in enumerate(report["rows"], start=2):
            t, o = row["team"], row["opp"]
            data.append([row["label"], _shots(t), t["makes"], _pps(t), _shots(o), o["makes"], _pps(o)])
            if t["small_sample"]:
                small.append(("BACKGROUND", (1, i), (3, i), SMALL_FILL))
            if o["small_sample"]:
                small.append(("BACKGROUND", (4, i), (6, i), SMALL_FILL))
        tt, ot = report["totals"]["team"], report["totals"]["opp"]
        data.append(["All zones", _shots(tt), tt["makes"], _pps(tt), _shots(ot), ot["makes"], _pps(ot)])
        table = Table(data, colWidths=[150] + [55] * 6)
        table.setStyle(TableStyle([
            ("SPAN", (1, 0), (3, 0)),
            ("SPAN", (4, 0), (6, 0)),
            ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 1), colors.white),
            ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            *small,
        ]))
        story.extend([
            table,
            Spacer(1, 8),
            Paragraph(escape(
                f"* Fewer than {report['small_sample_threshold']} shots: a small sample, so treat its rate with caution."
            ), styles["Normal"]),
        ])
    if report.get("dataset_version"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(escape(f"Data version: {report['dataset_version']}"), styles["Normal"]))
    doc.build(story)
    return buffer.getvalue()
