"""Analyst Synergy reload keeps the old data unless the new file passes (SCRUM-490)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from application.api_coordination.data_quality_endpoints import EXPECTED_SEASONS, EXPECTED_TEAMS
from application.data_reload_services.synergy_reload import (
    DataReloadError,
    ReloadInProgress,
    SynergyReloadService,
)
from domain.baseline_recommendation import BaselineRecommender

DATA = Path(__file__).resolve().parent.parent / "data"
SYNERGY = DATA / "synergy_playtypes_2019_2025_players.csv"
PREDICTIONS = DATA / "ml_offense_ppp_predictions.csv"


@pytest.fixture(scope="module")
def loaded():
    return BaselineRecommender(str(SYNERGY))


@pytest.fixture
def rec(loaded):
    # A fresh holder sharing the loaded tables, so tests can swap them freely.
    holder = BaselineRecommender.__new__(BaselineRecommender)
    holder.adopt(loaded)
    return holder


def service(rec, calls, predictions=PREDICTIONS, loader=BaselineRecommender):
    return SynergyReloadService(
        rec, predictions, EXPECTED_SEASONS, EXPECTED_TEAMS,
        on_reloaded=lambda r, p: calls.append((r, p)), loader=loader,
    )


def test_successful_reload_swaps_tables_and_reports_versions(rec, loaded):
    calls = []
    before = rec.raw_df
    result = service(rec, calls).reload()
    assert rec.raw_df is not before
    assert len(rec.raw_df) == len(loaded.raw_df)
    assert result["status"] == "reloaded" and result["source_file"] == SYNERGY.name
    assert result["dataset_version"] == result["previous_version"] == loaded.raw_df.attrs["dataset_version"]
    assert result["changed"] is False and result["quality_status"] == "pass"
    assert result["seasons"] == list(EXPECTED_SEASONS)
    assert len(calls) == 1 and calls[0][0] is rec and calls[0][1] is not None


def test_missing_file_keeps_previous_data(rec, tmp_path):
    calls = []
    before = rec.raw_df
    rec.source_path = tmp_path / "missing.csv"
    with pytest.raises(DataReloadError, match="missing.csv was not found"):
        service(rec, calls).reload()
    assert rec.raw_df is before and calls == []


def test_unreadable_file_keeps_previous_data(rec, tmp_path):
    calls = []
    before = rec.raw_df
    broken = tmp_path / "synergy.csv"
    broken.write_text("SEASON,TEAM\n2024-25\n", encoding="utf-8")
    rec.source_path = broken
    with pytest.raises(DataReloadError, match="synergy.csv could not be read"):
        service(rec, calls).reload()
    assert rec.raw_df is before and calls == []


def test_file_failing_quality_checks_keeps_previous_data(rec, loaded):
    calls = []
    before = rec.raw_df
    raw = loaded.raw_df[loaded.raw_df["TEAM_ABBREVIATION"] != "TOR"]
    no_toronto = SimpleNamespace(raw_df=raw, team_df=loaded.team_df, league_df=loaded.league_df, source_path=SYNERGY)
    with pytest.raises(DataReloadError, match="failed its data checks"):
        service(rec, calls, loader=lambda _path: no_toronto).reload()
    assert rec.raw_df is before and calls == []


def test_predictions_from_another_source_keep_previous_data(rec, tmp_path):
    calls = []
    before = rec.raw_df
    stale = tmp_path / "predictions.csv"
    stale.write_text(
        "SEASON,TEAM_ABBREVIATION,PLAY_TYPE,PPP_ML,SOURCE_DATASET_VERSION,AGGREGATION_VERSION,SEASON_PHASE\n"
        "2024-25,TOR,Cut,1.2,sha256:" + "0" * 64 + ",synergy_regular_season_v2,regular_season\n",
        encoding="utf-8",
    )
    with pytest.raises(DataReloadError, match="Rebuild them offline"):
        service(rec, calls, predictions=stale).reload()
    assert rec.raw_df is before and calls == []


def test_no_saved_predictions_still_reloads(rec, tmp_path):
    calls = []
    service(rec, calls, predictions=tmp_path / "none.csv").reload()
    assert calls[0][1] is None


def test_second_reload_waits_for_the_first(rec):
    reloader = service(rec, [])
    reloader._lock.acquire()
    try:
        with pytest.raises(ReloadInProgress):
            reloader.reload()
    finally:
        reloader._lock.release()
