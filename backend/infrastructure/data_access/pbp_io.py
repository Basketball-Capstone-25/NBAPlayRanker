"""Cached parquet reader utilities for PBP data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .pbp_constants import CANONICAL_PARQUET
from .pbp_loader import get_pbp_canonical_df

@lru_cache(maxsize=4)
def _read_parquet_full(path_str: str, mtime_ns: int) -> pd.DataFrame:
    # do not mutate it. We return copies when subsetting columns.
    return pd.read_parquet(path_str)

def read_parquet_cached(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Cached parquet read."""
    p = Path(path)

    # Canonical parquet: use the existing loader cache
    try:
        if p.resolve() == CANONICAL_PARQUET.resolve():
            df = get_pbp_canonical_df(ensure=True, force_rebuild=False)
        else:
            st = p.stat()
            df = _read_parquet_full(str(p), st.st_mtime_ns)
    except FileNotFoundError:
        raise
    except Exception:
        # Re-raise any other errors as-is to keep debugging clear
        raise

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Parquet missing expected columns {missing}. Available: {list(df.columns)}")
        return df[columns].copy()

    return df
