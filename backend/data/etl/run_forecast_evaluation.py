"""Write the season-ahead PPP report and the exact rows used for comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import numpy as np
import pandas as pd
import sklearn

from domain.statistical_analysis.forecast_evaluation import evaluate_forecasts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BACKEND / "data/synergy_playtypes_2019_2025_players.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/iteration-15/forecast")
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    report, rows = evaluate_forecasts(raw)
    source_files = [
        Path(__file__).resolve(),
        BACKEND / "domain/statistical_analysis/forecast_evaluation.py",
        BACKEND / "domain/statistical_analysis/calibration.py",
    ]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        changed = subprocess.check_output(["git", "status", "--porcelain", "--", *[str(p) for p in source_files]], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, changed = None, "unknown"
    report["provenance"] = {
        "git_head_at_run": revision,
        "evaluation_sources_uncommitted_at_run": bool(changed),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "dataset": args.data.name,
        "dataset_sha256": sha256(args.data),
        "raw_rows": len(raw),
        "regular_season_offense_player_rows": int((raw["SEASON_ID"].astype(str).str.fullmatch(r"2\d{4}") & raw["TYPE_GROUPING"].eq("Offensive")).sum()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "command": "python backend/data/etl/run_forecast_evaluation.py",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    rows.to_csv(predictions_path, index=False, float_format="%.12g")
    report["provenance"]["predictions_sha256"] = sha256(predictions_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Final test: {report['final_test']['target_season']}, {len(rows)} rows")
    for name, result in report["final_test"]["metrics"].items():
        print(f"{name}: MAE {result['mae_ppp']:.6f}, RMSE {result['rmse_ppp']:.6f}, bias {result['bias_ppp']:+.6f} PPP")
    print(f"Saved report.json and predictions.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
