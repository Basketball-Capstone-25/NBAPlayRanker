"""Download the pinned public hoopR sources, refusing changed release assets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = item["name"]
        if Path(name).name != name or not name.endswith(".parquet"):
            raise ValueError("Invalid source filename")
        expected_url = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_pbp/" + name
        if item["url"] != expected_url:
            raise ValueError("Unexpected source provider")
        path = args.output / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(expected_url, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > item["bytes"]:
                        raise ValueError(f"Source size changed: {name}")
                    output.write(chunk)
            if temporary.stat().st_size != item["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Source hash changed: {name}; review the new release before using it")
            temporary.replace(path)
            item["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Existing source does not match: {name}")
        print(name)
    (args.output / "download-manifest.json").write_text(json.dumps(items, indent=2))


if __name__ == "__main__":
    main()
