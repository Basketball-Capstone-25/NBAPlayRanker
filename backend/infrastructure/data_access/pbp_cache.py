"""Play-by-play cache utilities."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of a source file, used to tell whether a derived cache is still valid."""
    schema_version: str
    source_path: str
    source_mtime_ns: int
    source_size: int
    source_sha256: str | None = None

    @property
    def id(self) -> str:
        if self.source_sha256:
            return f"{self.schema_version}|{self.source_sha256}"
        return f"{self.schema_version}|{self.source_path}|{self.source_mtime_ns}|{self.source_size}"


def ensure_dir(path: Path) -> None:
    """Create a folder and its parents if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


def fingerprint_file(path: Path, *, schema_version: str) -> FileFingerprint:
    """Fingerprint a source file for the given cache schema version."""
    p = Path(path)
    st = p.stat()
    from .dataset_metadata import source_version
    return FileFingerprint(
        schema_version=str(schema_version),
        source_path=str(p.resolve()),
        source_mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        source_size=int(st.st_size),
        source_sha256=source_version(p),
    )


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, or None if it is missing or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON through a temporary file so readers never see a half-written file."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, p)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def write_parquet_atomic(df: Any, path: Path) -> None:
    """Write Parquet through a temporary file, then swap it in."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, p)


def cache_valid(meta_path: Path, expected: FileFingerprint) -> bool:
    """True when the cache metadata matches the expected source fingerprint."""
    meta = read_json(meta_path)
    if not meta:
        return False
    stored = meta.get("fingerprint")
    if not isinstance(stored, dict):
        return False
    try:
        fp = FileFingerprint(
            schema_version=str(stored["schema_version"]),
            source_path=str(stored["source_path"]),
            source_mtime_ns=int(stored["source_mtime_ns"]),
            source_size=int(stored["source_size"]),
            source_sha256=stored.get("source_sha256"),
        )
    except Exception:
        return False
    return fp.id == expected.id


def build_meta(*, fingerprint: FileFingerprint, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Metadata to store next to a cache: its source fingerprint plus extra fields."""
    out: Dict[str, Any] = {"fingerprint": asdict(fingerprint)}
    if extra:
        out.update(extra)
    return out
