"""Evaluate pre-game shot forecasts and save results outside the code checkout."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import pandas as pd
import sklearn

from domain.shot_analysis.shot_forecast_features import build_pregame_rows
from domain.shot_analysis.shot_forecast_evaluation import evaluate_pregame_forecasts
from infrastructure.data_access.shot_forecast_source import file_sha256, load_shot_forecast_source


def run(clean_path: Path, raw_dir: Path, output_dir: Path) -> dict:
    """Evaluate pre-game shot forecasts from the pinned shot source and return the report."""
    destination = output_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Save evaluation results outside the code checkout.")
    shots, provenance = load_shot_forecast_source(clean_path, raw_dir)
    rows, history_coverage = build_pregame_rows(shots)
    report, predictions = evaluate_pregame_forecasts(rows, history_coverage)
    source_files = [Path(__file__).resolve(), BACKEND / "infrastructure/data_access/shot_forecast_source.py",
        *sorted((BACKEND / "domain/shot_analysis").glob("shot_forecast_*.py"))]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        changed = bool(subprocess.check_output(["git", "status", "--porcelain", "--",
            *[str(path) for path in source_files]], cwd=ROOT, text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, changed = None, None
    report["provenance"] = {
        **provenance, "git_head_at_run": revision, "evaluation_sources_uncommitted_at_run": changed,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "predictions.csv"
    predictions.to_csv(csv_path, index=False, float_format="%.12g", date_format="%Y-%m-%d")
    report["provenance"]["predictions_sha256"] = file_sha256(csv_path)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    """Command-line entry: run the evaluation and save results outside the code checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=BACKEND / "data/pbp/shots_clean.parquet")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing the pinned original Parquet sources.")
    parser.add_argument("--output-dir", type=Path, required=True, help="External folder for report.json and predictions.csv.")
    args = parser.parse_args()
    report = run(args.clean, args.raw_dir, args.output_dir)
    print(f"Held-out season: {report['holdout']['season']}; {report['holdout']['shots']:,} shots")
    for name, metrics in report["holdout"]["metrics"].items():
        print(f"{name}: RMSE {metrics['rmse_points']:.6f}, MAE {metrics['mae_points']:.6f}, bias {metrics['bias_points']:+.6f} points/shot")
    print(f"Saved report.json and predictions.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
