from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas

from domain.baseline_recommendation import ShotBaselineRecommender
from domain.shot_analysis import CLEAN_PARQUET


_SHOT_REC: Optional[ShotBaselineRecommender] = None
_SHOTS_DF: Optional[pd.DataFrame] = None


def _get_shot_rec() -> ShotBaselineRecommender:
    global _SHOT_REC
    if _SHOT_REC is None:
        _SHOT_REC = ShotBaselineRecommender()
    return _SHOT_REC


def _get_shots_df() -> pd.DataFrame:
    global _SHOTS_DF
    if _SHOTS_DF is None:
        if not CLEAN_PARQUET.exists():
            raise FileNotFoundError(
                f"shots_clean.parquet not found: {CLEAN_PARQUET}. Run backend/data/etl/build_pbp_pipeline.py"
            )
        _SHOTS_DF = pd.read_parquet(CLEAN_PARQUET)
    return _SHOTS_DF


_ACCENT = colors.HexColor("#1a3c5e")
_HEADER_BG = colors.HexColor("#1a3c5e")
_HEADER_FG = colors.white
_ALT_ROW = colors.HexColor("#f4f6f8")


def _fmt_num(x: object, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _best_epa(row: Dict[str, object]) -> str:
    """Return the first non-None EPA value from the preferred cascade."""
    for key in ("EPA_PRED", "EPA_OFF_SHRUNK", "EPA_OFF"):
        val = row.get(key)
        if val is not None:
            return _fmt_num(val)
    return "—"


def _table_from_rows(title: str, rows: List[List[str]]) -> Table:
    col_w = [1.8 * inch, 0.8 * inch, 0.7 * inch]
    data = [
        [title, "", ""],          # row 0 — merged title
        ["Name", "EPA", "Att."],  # row 1 — column headers
    ] + rows

    table = Table(data, colWidths=col_w)
    n_rows = len(data)
    style_cmds = [
        # Title row
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        # Column header row
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8ecf0")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        # Data rows
        ("FONTNAME", (0, 2), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 2), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 2), (-1, -1), 3),
        ("TOPPADDING", (0, 2), (-1, -1), 3),
        # Right-align numeric columns everywhere except title row
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        # Outer box and light inner grid
        ("BOX", (0, 0), (-1, -1), 0.5, _ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, _ACCENT),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.grey),
        ("INNERGRID", (0, 2), (-1, -1), 0.25, colors.HexColor("#dde1e6")),
    ]
    # Alternate row shading for data rows
    for i in range(2, n_rows):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _ALT_ROW))
    table.setStyle(TableStyle(style_cmds))
    return table


def build_shotplan_pdf(
    *,
    season: str,
    our: str,
    opp: str,
    k: int,
    w_off: float,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
) -> Tuple[io.BytesIO, str]:
    # Rankings
    rec = _get_shot_rec()
    w_def = float(1.0 - w_off)
    result = rec.rank(season=season, our_team=our, opp_team=opp, k=int(k), w_off=float(w_off))

    top_shot_types = result.get("top_shot_types", [])
    top_zones = result.get("top_zones", [])

    # Heatmap image (lazy import)
    try:
        from infrastructure.visualization_and_export.viz_shot_heatmap import render_shot_heatmap_png
    except Exception as e:
        raise RuntimeError(f"viz_shot_heatmap import failed: {e}")

    shots_df = _get_shots_df()
    title = f"{our} vs {opp} • {season}"
    png_bytes = render_shot_heatmap_png(
        shots_df=shots_df,
        season=season,
        our_team=our,
        opp_team=opp,
        shot_type=shot_type,
        zone=zone,
        title=title,
    )

    # Build PDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter
    margin = 0.6 * inch
    usable_w = page_w - 2 * margin
    y = page_h - margin

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(_ACCENT)
    c.drawString(margin, y, "Shot Plan")
    y -= 0.25 * inch

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.black)
    c.drawString(margin, y, f"{our} vs {opp}  ·  {season}")
    y -= 0.18 * inch

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(
        margin, y,
        f"Offense weight {w_off:.0%}  ·  Defense weight {w_def:.0%}  ·  Top-K {int(k)}",
    )
    c.setFillColor(colors.black)
    y -= 0.30 * inch

    c.setStrokeColor(_ACCENT)
    c.setLineWidth(1)
    c.line(margin, y, page_w - margin, y)
    y -= 0.25 * inch

    def _rows_from(items: List[Dict[str, object]], label_key: str) -> List[List[str]]:
        rows = []
        for r in items[: int(k)]:
            label = str(r.get(label_key, "—"))
            epa = _best_epa(r)
            attempts = _fmt_num(r.get("attempts_OFF"), digits=0)
            rows.append([label, epa, attempts])
        if not rows:
            rows = [["No data", "—", "—"]]
        return rows

    shot_type_table = _table_from_rows(
        "Top Shot Types", _rows_from(top_shot_types, "SHOT_TYPE"),
    )
    zone_table = _table_from_rows(
        "Top Zones", _rows_from(top_zones, "ZONE"),
    )

    # Each table is 3.3" wide; gap 0.5" → total 7.1" fits in 7.3" usable
    table_w = 3.3 * inch
    gap = 0.5 * inch
    tw, th = shot_type_table.wrap(table_w, 3 * inch)       # measure height
    zw, zh = zone_table.wrap(table_w, 3 * inch)
    table_h = max(th, zh)

    shot_type_table.drawOn(c, margin, y - table_h)
    zone_table.drawOn(c, margin + table_w + gap, y - table_h)
    y -= table_h + 0.30 * inch

    c.setStrokeColor(colors.HexColor("#cccccc"))
    c.setLineWidth(0.5)
    c.line(margin, y, page_w - margin, y)
    y -= 0.22 * inch

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(_ACCENT)
    c.drawString(margin, y, "Court View")
    c.setFillColor(colors.black)
    y -= 0.20 * inch

    img_reader = ImageReader(io.BytesIO(png_bytes))
    img_w = 5.0 * inch
    img_h = 4.2 * inch
    img_x = margin + (usable_w - img_w) / 2        # centre horizontally
    img_y = y - img_h
    c.drawImage(
        img_reader, img_x, img_y,
        width=img_w, height=img_h,
        preserveAspectRatio=True, anchor="c",
    )
    y = img_y - 0.25 * inch

    c.setStrokeColor(colors.HexColor("#cccccc"))
    c.setLineWidth(0.5)
    c.line(margin, y, page_w - margin, y)
    y -= 0.18 * inch

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(_ACCENT)
    c.drawString(margin, y, "Shrinkage Note")
    c.setFillColor(colors.black)
    y -= 0.14 * inch

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#444444"))
    note = (
        "EPA is shrunk toward league baselines using reliability weights "
        "(log1p(attempts)) to reduce small-sample noise. "
        "Matchup EPA blends our offense with opponent defense."
    )
    text = c.beginText(margin, y)
    text.setLeading(11)
    words = note.split(" ")
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if c.stringWidth(candidate, "Helvetica", 8) <= usable_w:
            line = candidate
        else:
            text.textLine(line)
            line = w
    if line:
        text.textLine(line)
    c.drawText(text)
    c.setFillColor(colors.black)

    # Footer
    c.setFont("Helvetica-Oblique", 7)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawRightString(page_w - margin, margin * 0.6, "Generated by NBA Play Ranker  ·  Basketball Strategy Analysis (Capstone)")
    c.setFillColor(colors.black)
    c.showPage()
    c.save()

    buf.seek(0)
    filename = f"shotplan_{season}_{our}_vs_{opp}.pdf".replace(" ", "_")
    return buf, filename
