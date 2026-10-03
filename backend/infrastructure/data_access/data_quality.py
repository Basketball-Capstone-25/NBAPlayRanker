"""Data quality checks for the loaded Synergy/PBP datasets (SCRUM-478).

Implements the "Run Data Quality Checks" use case: schema, completeness,
consistency (duplicates) and coverage checks, summarised in one report so
malformed data is flagged instead of silently ranked on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

PASS = "pass"
FAIL = "fail"
NOT_RUN = "not_run"

SYNERGY_REQUIRED_COLUMNS: List[str] = [
    "SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE", "TYPE_GROUPING",
    "POSS", "PPP", "PTS",
]
SYNERGY_KEY_FIELDS: List[str] = ["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE", "POSS", "PPP"]
# SEASON_ID separates regular-season (2xxxx) from playoff (4xxxx) rows.
SYNERGY_ROW_KEY: List[str] = [
    "SEASON_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "PLAY_TYPE", "TYPE_GROUPING",
]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataQualityReport:
    status: str
    safe_for_use: bool
    checks: List[CheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _missing(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


class DataQualityChecker:
    """Each check returns a CheckResult; a check that cannot run says why."""

    def check_schema(self, df: pd.DataFrame, required_columns: Sequence[str]) -> CheckResult:
        missing = _missing(df, required_columns)
        return CheckResult(
            "schema",
            FAIL if missing else PASS,
            {"required_columns": list(required_columns), "missing_columns": missing},
        )

    def check_completeness(self, df: pd.DataFrame, key_fields: Sequence[str]) -> CheckResult:
        missing = _missing(df, key_fields)
        if missing:
            return CheckResult("completeness", NOT_RUN, {"reason": f"missing columns: {missing}"})
        null_counts = {
            column: int(count)
            for column, count in df[list(key_fields)].isna().sum().items()
            if count > 0
        }
        return CheckResult(
            "completeness",
            FAIL if null_counts else PASS,
            {"rows": int(len(df)), "null_counts": null_counts},
        )

    def check_duplicates(self, df: pd.DataFrame, key_fields: Sequence[str]) -> CheckResult:
        missing = _missing(df, key_fields)
        if missing:
            return CheckResult("duplicates", NOT_RUN, {"reason": f"missing columns: {missing}"})
        duplicate_rows = int(df.duplicated(subset=list(key_fields), keep="first").sum())
        return CheckResult(
            "duplicates",
            FAIL if duplicate_rows else PASS,
            {"key_fields": list(key_fields), "duplicate_rows": duplicate_rows},
        )

    def check_coverage(
        self,
        df: pd.DataFrame,
        seasons: Sequence[str],
        teams: Sequence[str],
    ) -> CheckResult:
        missing = _missing(df, ["SEASON", "TEAM_ABBREVIATION"])
        if missing:
            return CheckResult("coverage", NOT_RUN, {"reason": f"missing columns: {missing}"})
        if not seasons or not teams:
            return CheckResult("coverage", NOT_RUN, {"reason": "no expected seasons or teams supplied"})
        present = set(
            zip(df["SEASON"].astype(str), df["TEAM_ABBREVIATION"].astype(str))
        )
        missing_pairs = sorted(
            f"{season}:{team}"
            for season in seasons
            for team in teams
            if (str(season), str(team)) not in present
        )
        return CheckResult(
            "coverage",
            FAIL if missing_pairs else PASS,
            {
                "expected_seasons": len(seasons),
                "expected_teams": len(teams),
                "missing_season_team_pairs": missing_pairs,
            },
        )

    def run_data_quality_checks(
        self,
        df: pd.DataFrame,
        *,
        required_columns: Sequence[str] = SYNERGY_REQUIRED_COLUMNS,
        key_fields: Sequence[str] = SYNERGY_KEY_FIELDS,
        row_key: Sequence[str] = SYNERGY_ROW_KEY,
        seasons: Optional[Sequence[str]] = None,
        teams: Optional[Sequence[str]] = None,
    ) -> DataQualityReport:
        checks = [
            self.check_schema(df, required_columns),
            self.check_completeness(df, key_fields),
            self.check_duplicates(df, row_key),
            self.check_coverage(df, list(seasons or []), list(teams or [])),
        ]
        failed = any(check.status == FAIL for check in checks)
        not_run = any(check.status == NOT_RUN for check in checks)
        status = FAIL if failed else (NOT_RUN if not_run else PASS)
        return DataQualityReport(status=status, safe_for_use=status == PASS, checks=checks)


def run_data_quality_checks(df: pd.DataFrame, **kwargs: Any) -> DataQualityReport:
    return DataQualityChecker().run_data_quality_checks(df, **kwargs)
