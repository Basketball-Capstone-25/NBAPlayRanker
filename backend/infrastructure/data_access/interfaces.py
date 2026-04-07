"""Data access interfaces – public contracts for the Data Access subsystem."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import pandas as pd


class IGetShotMetaOptions(Protocol):
    """Return dropdown-filter options (seasons, teams, shot types, zones)."""

    def __call__(self) -> Dict[str, Any]: ...


class IGetShotsJson(Protocol):
    """Return a filtered JSON preview of shot rows."""

    def __call__(
        self,
        *,
        season: str,
        team: str,
        opp: Optional[str],
        shot_type: Optional[str],
        zone: Optional[str],
        limit: int,
    ) -> Dict[str, Any]: ...


class IGetShotsCsvResponse(Protocol):
    """Return a streaming CSV response of filtered shot rows."""

    def __call__(
        self,
        *,
        season: str,
        team: str,
        opp: Optional[str],
        shot_type: Optional[str],
        zone: Optional[str],
        limit: int,
    ) -> Any: ...  # StreamingResponse at runtime


class IRenderPbpHeatmapPng(Protocol):
    """Render a PBP shot heatmap and return raw PNG bytes."""

    def __call__(
        self,
        *,
        season: str,
        team: str,
        opp: str,
        shot_type: Optional[str] = None,
        zone: Optional[str] = None,
        max_shots: int = 35_000,
    ) -> bytes: ...


class IRenderPbpHeatmapBase64(Protocol):
    """Render a PBP shot heatmap and return a ``{caption, image_base64}`` dict."""

    def __call__(
        self,
        *,
        season: str,
        our: str,
        opp: str,
        shot_type: Optional[str] = None,
        zone: Optional[str] = None,
        max_shots: int = 35_000,
    ) -> Dict[str, Any]: ...


class IFingerprintFile(Protocol):
    """Build an immutable fingerprint of a source file for cache validation."""

    def __call__(self, path: Path, *, schema_version: str) -> Any: ...


class ICacheValid(Protocol):
    """Check whether a cached artifact is still valid against a fingerprint."""

    def __call__(self, meta_path: Path, expected: Any) -> bool: ...


class IBuildMeta(Protocol):
    """Build a metadata dict from a fingerprint, optionally with extras."""

    def __call__(
        self,
        *,
        fingerprint: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


class IReadJson(Protocol):
    """Read a JSON file and return its contents, or ``None`` on any error."""

    def __call__(self, path: Path) -> Optional[Dict[str, Any]]: ...


class IWriteJsonAtomic(Protocol):
    """Atomically write a dict to a JSON file."""

    def __call__(self, path: Path, data: Dict[str, Any]) -> None: ...


class IGetPbpCanonicalDf(Protocol):
    """Return the cached canonical PBP shots DataFrame."""

    def __call__(
        self,
        *,
        ensure: bool = True,
        force_rebuild: bool = False,
    ) -> pd.DataFrame: ...


class ILoadCanonicalDf(Protocol):
    """Load and return the canonical shots DataFrame from disk."""

    def __call__(self, *, force_rebuild: bool = False) -> pd.DataFrame: ...
