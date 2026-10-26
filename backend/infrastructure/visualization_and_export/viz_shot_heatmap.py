from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional, Tuple
from threading import RLock
from functools import wraps

_RENDER_LOCK = RLock()

def _serialized_render(function):
    @wraps(function)
    def render(*args, **kwargs):
        # Matplotlib has process-wide drawing state. Keep concurrent requests
        # from changing that state while another image is being rendered.
        with _RENDER_LOCK:
            return function(*args, **kwargs)
    return render

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
    df = shots_df.loc[(shots_df["SEASON_STR"] == season) & (shots_df["TEAM_ABBR"] == our_team)]
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

        if "HOME_FLAG" in df:
            # hoopR home shots attack +X and away shots -X. Rotate by the
            # attacking basket so a heave keeps its actual distance/direction.
            orientation = np.where(df["HOME_FLAG"].to_numpy(dtype=bool)[ok], 1.0, -1.0) * side_sign
            x *= orientation
            y *= orientation
        else:
            mask = (x * side_sign) < 0
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


def heatmap_plot_data(df: pd.DataFrame):
    """Return only usable locations on the attacking half of the physical court."""
    finite = np.isfinite(pd.to_numeric(df["X"], errors="coerce")) & np.isfinite(pd.to_numeric(df["Y"], errors="coerce"))
    valid = finite & df.get("LOCATION_VALID", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    usable = df.loc[valid]
    class PhysicalAxis:
        def get_xlim(self):
            return (0.0, 47.0)
    class PhysicalCourt:
        court_params = {"basket_center_to_baseline": 5.25}
    x, y = _coords_to_sportypy_halfcourt(df=usable, ax=PhysicalAxis(), court=PhysicalCourt())
    on = (x >= 0.0) & (x <= 47.0) & (np.abs(y) <= 25.0)
    counts = {
        "n_shots_excluded_location": int((~valid).sum()),
        "n_shots_outside_halfcourt": int((~on).sum()),
    }
    return usable.iloc[np.flatnonzero(on)], x[on], y[on], counts


@_serialized_render
def render_shot_heatmap_png(
    *,
    shots_df: pd.DataFrame,
    season: str,
    our_team: str,
    opp_team: str,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
    title: Optional[str] = None,
    matched_count: Optional[int] = None,
) -> bytes:
    _patch_sportypy_circle(400)
    df = _filter_shots(shots_df, season=season, our_team=our_team,
                       opp_team=opp_team, shot_type=shot_type, zone=zone)
    _, x, y, counts = heatmap_plot_data(df)
    court = NBACourt()
    fig, ax = plt.subplots(figsize=(7, 6))
    court.draw(ax=ax, display_range="offense")
    # SportyPy adds decoration padding. The data grid uses physical court bounds.
    ax.set_xlim(-1, 48)
    ax.set_ylim(-26, 26)
    if x.size:
        hb = ax.hexbin(x, y, gridsize=35, extent=(0, 47, -25, 25),
                       cmap="Reds", mincnt=1, alpha=0.85, linewidths=0,
                       vmin=0, zorder=30)
        colorbar = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Attempts per hexagon", fontsize=9)
        from matplotlib.ticker import MaxNLocator
        colorbar.locator = MaxNLocator(nbins=5, integer=True)
        colorbar.update_ticks()
    else:
        message = "No shots found for filters" if df.empty and not matched_count else "No usable locations in the displayed half court"
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=10)
    ax.set_title(title or "Shot Heatmap", fontsize=11, weight="bold")
    ax.axis("off")
    foot = "Attempt counts (makes + misses). Color scale varies by chart."
    if counts["n_shots_excluded_location"]:
        foot += f"\n{counts['n_shots_excluded_location']:,} uncertain/missing locations excluded."
    if counts["n_shots_outside_halfcourt"]:
        foot += f"\n{counts['n_shots_outside_halfcourt']:,} attempts outside the displayed half court."
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.14, top=0.84)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
