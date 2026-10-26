"""Print the shot column mapping without training or changing the input file."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pyarrow.parquet as parquet
import pandas as pd
from domain.shot_analysis.shot_etl import CLEAN_PARQUET
from domain.shot_analysis.shot_ml_models import shot_schema_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=CLEAN_PARQUET)
    args = parser.parse_args()
    columns = parquet.ParquetFile(args.path).schema_arrow.names
    report = shot_schema_report(pd.DataFrame(columns=columns))
    print(json.dumps(report, indent=2))
    return 1 if report["missing_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
