from __future__ import annotations

from io import BytesIO
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge

from sportypy.surfaces.basketball import NBACourt

from infrastructure.visualization_and_export.viz_shot_heatmap import (
    _patch_sportypy_circle,
    png_bytes_to_base64,
)


PLAYTYPE_ZONES: Dict[str, Dict[str, float]] = {
    "Spotup": {"corner3": 1.0, "arc3": 0.8},
    "Cut": {"rim": 1.0},
    "Transition": {"rim": 0.9, "corner3": 0.6},
    "Isolation": {"mid": 0.8, "rim": 0.4},
    "OffScreen": {"arc3": 0.9, "mid": 0.4},
    "HandOff": {"arc3": 0.7, "mid": 0.4},
    "PostUp": {"paint": 1.0},
    "P&R Ball Handler": {"rim": 0.7, "mid": 0.5, "arc3": 0.4},
    "P&R Roll Man": {"rim": 1.0, "paint": 0.6},
}


def zones_for_playtype(play_type: str) -> Dict[str, float]:
    # Robust matching: Synergy labels vary ("Spot Up" vs "Spotup" vs "Spot-Up")
    def _norm(s: str) -> str:
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    pt = _norm(play_type)

    for key, zones in PLAYTYPE_ZONES.items():
        if _norm(key) in pt:
            return zones
    return {"rim": 0.6, "arc3": 0.4}  # fallback


def render_playtype_zone_png(play_type: str, title: str) -> bytes:
    _patch_sportypy_circle(400)
    zones = zones_for_playtype(play_type)

    court = NBACourt()
    fig, ax = plt.subplots(figsize=(7, 6))
    court.draw(ax=ax, display_range="offense")

    # SportyPy stores dimensions in court.court_params (dict)
    cp = getattr(court, "court_params", {}) or {}

    def _scalar(v, default: float) -> float:
        # SportyPy stores some params as lists; normalize to a single float
        if v is None:
            return float(default)
        if isinstance(v, (list, tuple)):
            return float(v[0]) if len(v) else float(default)
        return float(v)

    court_length = _scalar(cp.get("court_length"), 94.0)
    basket_center_to_baseline = _scalar(cp.get("basket_center_to_baseline"), 5.25)
    court_width = _scalar(cp.get("court_width"), 50.0)
    lane_length = _scalar(cp.get("lane_length"), 19.0)
    lane_width = _scalar(cp.get("lane_width"), 16.0)

    # In SportyPy, x=0 is midcourt; offense baseline is at +court_length/2
    baseline_x = court_length / 2.0  # ~47
    hoop_x = baseline_x - basket_center_to_baseline  # ~41.75

    def add_zone(patch, w):
        patch.set_alpha(0.15 + 0.35 * float(w))
        patch.set_facecolor((1, 0, 0))
        patch.set_edgecolor("none")
        # SportyPy court elements can have high z-order; force overlays on top.
        try:
            patch.set_zorder(20)
        except Exception:
            pass
        try:
            patch.set_linewidth(0)
        except Exception:
            pass
        ax.add_patch(patch)

    if "rim" in zones:
        add_zone(Circle((hoop_x, 0), radius=4.0), zones["rim"])

    if "paint" in zones:
        add_zone(
            Rectangle((baseline_x - lane_length, -lane_width / 2.0), lane_length, lane_width),
            zones["paint"],
        )

    if "mid" in zones:
        # Keep the "mid" band within the drawable court width (prevents spill beyond sidelines)
        y_max = court_width / 2.0
        mid_half = min(19.0, max(0.0, y_max - 1.0))
        add_zone(
            Rectangle((baseline_x - lane_length, -mid_half), lane_length, 2.0 * mid_half),
            zones["mid"],
        )

    if "corner3" in zones:
        # Use the court width instead of hard-coded y values so the bands
        # don't get clipped or look "off" relative to the sidelines.
        y_max = court_width / 2.0
        band_h = min(6.0, y_max)
        y_top = y_max - band_h
        x0 = baseline_x - 14
        add_zone(Rectangle((x0, y_top), 14, band_h), zones["corner3"])
        add_zone(Rectangle((x0, -y_max), 14, band_h), zones["corner3"])

    if "arc3" in zones:
        # ✅ FIX: The 3pt arc highlight was pointing the wrong way (toward the baseline),
        # which makes it appear "behind the rim" and get clipped. The arc must point
        # toward midcourt, i.e. centered on 180° for a +x baseline.
        #
        # Also: highlight should sit ON the 3pt line, so we draw a thin band centered
        # around the 3pt arc radius (default NBA ~23.75 ft).
        x0, x1 = ax.get_xlim()
        baseline_side = (x1 if abs(x1) > abs(x0) else x0)
        if baseline_side >= 0:
            theta1, theta2 = 112, 248  # centered on 180° (toward midcourt)
        else:
            theta1, theta2 = -68, 68   # centered on 0° (toward midcourt on left baseline)

        three_pt_r = _scalar(cp.get("three_point_arc_radius"), 23.75)

        band_w = 2.0  # thin band around the line
        outer_r = three_pt_r + (band_w / 2.0)  # center the band on the line

        add_zone(
            Wedge((hoop_x, 0), r=outer_r, theta1=theta1, theta2=theta2, width=band_w),
            zones["arc3"],
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    buf = BytesIO()
    # bbox_inches="tight" can crash with some SportyPy artists
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
