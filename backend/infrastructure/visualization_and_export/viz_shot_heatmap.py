from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sportypy.surfaces.basketball import NBACourt
from sportypy._base_classes._base_feature import BaseFeature


def _patch_sportypy_circle(npoints: int = 400) -> None:
    def fast_create_circle(center=(0.0, 0.0), npoints=npoints, r=1.0, start=0.0, end=2.0):
        theta = np.linspace(start * np.pi, end * np.pi, npoints)
        x = center[0] + (r * np.cos(theta))
        y = center[1] + (r * np.sin(theta))
        return pd.DataFrame({"x": x, "y": y})

    BaseFeature.create_circle = staticmethod(fast_create_circle)


def png_bytes_to_base64(png: bytes) -> str:
    return base64.b64encode(png).decode("utf-8")


def _filter_shots(
    shots_df: pd.DataFrame,
    *,
    season: str,
    our_team: str,
    opp_team: str,
    shot_type: Optional[str],
    zone: Optional[str],
) -> pd.DataFrame:
    df = shots_df.copy()
    df = df[df["SEASON_STR"] == season]
    df = df[df["TEAM_ABBR"] == our_team]
    if "OPP_ABBR" in df.columns and opp_team:
        df = df[df["OPP_ABBR"] == opp_team]
    if shot_type:
        df = df[df["SHOT_TYPE"] == shot_type]
    if zone:
        df = df[df["ZONE"] == zone]
    return df


def _scalar(v, default: float) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        return float(v[0]) if len(v) else float(default)
    return float(v)


def _coords_to_sportypy_halfcourt(
    *,
    df: pd.DataFrame,
    ax,
    court: NBACourt,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert df["X"], df["Y"] into SportyPy half-court coordinates.

    Handles two common cases:
    (A) Already SportyPy-like: feet, midcourt origin (x ~ [-47,47], y ~ [-25,25])
    (B) NBA shotchart-like: inches, hoop origin (LOC_X lateral ~ [-250,250], LOC_Y depth ~ [0,470])
        We convert to feet and map to SportyPy:
           hoop_x ~= baseline_x - basket_center_to_baseline
           x_sporty = hoop_x - side_sign * depth_ft
           y_sporty = lateral_ft
    """
    x_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=float)

    # Drop NaNs early
    ok = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[ok]
    y_raw = y_raw[ok]

    if x_raw.size == 0:
        return x_raw, y_raw

    # Visible baseline side from SportyPy's drawn axis limits
    x0, x1 = ax.get_xlim()
    baseline_x = x0 if abs(x0) > abs(x1) else x1  # ~ +/-47
    side_sign = 1.0 if baseline_x >= 0 else -1.0

    cp = getattr(court, "court_params", {}) or {}
    basket_center_to_baseline = _scalar(cp.get("basket_center_to_baseline"), 5.25)
    hoop_x = baseline_x - side_sign * basket_center_to_baseline

    max_abs_x = float(np.nanmax(np.abs(x_raw)))
    max_abs_y = float(np.nanmax(np.abs(y_raw)))

    # Case (A): already in feet / midcourt space (rough bounds)
    # (gives ranges like x ~ [-47,47], y ~ [-25,25])
    if max_abs_x <= 60.0 and max_abs_y <= 35.0:
        x = x_raw.copy()
        y = y_raw.copy()

        # Ensure everything lands on the displayed half (rotate opposite end 180°)
        mask = (x * side_sign) < 0
        if mask.any():
            x[mask] = -x[mask]
            y[mask] = -y[mask]
        return x, y

    # Case (B): NBA shotchart style inches around hoop.
    # Detect which axis looks like "depth" (0..470 inches) vs "lateral" (-250..250).
    # Typical: X = lateral (LOC_X), Y = depth (LOC_Y)
    # Some datasets swap these; handle both.
    x_min, x_max = float(np.nanmin(x_raw)), float(np.nanmax(x_raw))
    y_min, y_max = float(np.nanmin(y_raw)), float(np.nanmax(y_raw))

    # Heuristic: depth is usually mostly non-negative and has larger magnitude (~400+)
    x_looks_depth = (x_max - x_min) > 300 and x_max > 200 and x_min > -50
    y_looks_depth = (y_max - y_min) > 300 and y_max > 200 and y_min > -50

    if y_looks_depth and not x_looks_depth:
        lateral_in = x_raw
        depth_in = y_raw
    elif x_looks_depth and not y_looks_depth:
        lateral_in = y_raw
        depth_in = x_raw
    else:
        # Fallback: assume canonical NBA shotchart mapping
        lateral_in = x_raw
        depth_in = y_raw

    # Inches -> feet
    lateral_ft = lateral_in / 12.0
    depth_ft = depth_in / 12.0

    # Map hoop-origin to SportyPy midcourt-origin halfcourt:
    # baseline side_sign decides which direction is "toward midcourt"
    x = hoop_x - side_sign * depth_ft
    y = lateral_ft

    return x.astype(float), y.astype(float)


def render_shot_heatmap_png(
    *,
    shots_df: pd.DataFrame,
    season: str,
    our_team: str,
    opp_team: str,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
    title: Optional[str] = None,
) -> bytes:
    _patch_sportypy_circle(400)

    df = _filter_shots(
        shots_df,
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        shot_type=shot_type,
        zone=zone,
    )

    court = NBACourt()
    fig, ax = plt.subplots(figsize=(7, 6))
    court.draw(ax=ax, display_range="offense")

    if df.empty:
        ax.set_title(title or "Shot Heatmap", fontsize=14, weight="bold")
        ax.text(
            0.5,
            0.5,
            "No shots found for filters",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    # ✅ Key fix: convert X/Y into SportyPy coordinates (feet + midcourt origin)
    x, y = _coords_to_sportypy_halfcourt(df=df, ax=ax, court=court)

    # If conversion produced nothing usable
    if x.size == 0:
        ax.set_title(title or "Shot Heatmap", fontsize=14, weight="bold")
        ax.text(
            0.5,
            0.5,
            "Shot coordinates missing/invalid",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    # Optional: drop points that are wildly off-court (helps if a few bad rows exist)
    # SportyPy halfcourt generally shows x in [min(xlim), max(xlim)] and y in about [-25, 25]
    x0, x1 = ax.get_xlim()
    xmin, xmax = (min(x0, x1) - 2.0), (max(x0, x1) + 2.0)
    on = (x >= xmin) & (x <= xmax) & (np.abs(y) <= 30.0)
    x = x[on]
    y = y[on]

    if x.size == 0:
        ax.set_title(title or "Shot Heatmap", fontsize=14, weight="bold")
        ax.text(
            0.5,
            0.5,
            "All shots landed off-court after normalization",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    # Plot heatmap on top of court
    # Use log bins only when we have enough shots; otherwise it can look “blank”
    use_log = x.size >= 250

    hb = ax.hexbin(
        x,
        y,
        gridsize=35,
        cmap="Reds",
        bins="log" if use_log else None,
        mincnt=1,
        alpha=0.85,
        linewidths=0,
    )
    # Ensure it draws above court artwork
    try:
        hb.set_zorder(30)
    except Exception:
        pass

    ax.set_title(title or "Shot Heatmap", fontsize=14, weight="bold")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
