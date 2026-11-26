"""Visualization and export interfaces."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import pandas as pd

class IRenderShotHeatmapPng(Protocol):
    """Render a shot-distribution heatmap and return raw PNG bytes."""

    def __call__(
        self,
        *,
        shots_df: pd.DataFrame,
        season: str,
        our_team: str,
        opp_team: str,
        shot_type: Optional[str] = None,
        zone: Optional[str] = None,
        title: Optional[str] = None,
    ) -> bytes:
        """Render the recorded shots that match the team, opponent and optional shot filters."""
        ...

class IRenderPlaytypeZonePng(Protocol):
    """Render a play-type zone court diagram and return PNG bytes."""

    def __call__(self, play_type: str, title: str) -> bytes:
        """Draw the configured zones for one play type under the requested title."""
        ...

class IPngBytesToBase64(Protocol):
    """Encode raw PNG bytes as base64 text."""

    def __call__(self, png: bytes) -> str:
        """Return base64 text suitable for the image field in a JSON response."""
        ...

class IBuildPlaytypeVizPdf(Protocol):
    """Build a full play-type visualization PDF report."""

    def __call__(
        self,
        play_type: str,
        season: str,
        our: str,
        opp: str,
        caption: str,
        png_bytes: bytes,
        w_off: float,
        w_def: float,
        k: int,
        team_names: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Package the court image with the selected matchup, ranking weights and caption."""
        ...
