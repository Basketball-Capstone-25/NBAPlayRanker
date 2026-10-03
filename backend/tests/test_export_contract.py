"""Unified export contract: one payload, consistent CSV/JSON/PDF output (SCRUM-484)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from application.api_coordination.export_contract import (
    CONTRACT_VERSION,
    ExportContract,
    ExportFormatUnavailable,
    ExportPayload,
)


def _payload() -> ExportPayload:
    return ExportPayload(
        report="baseline_rankings",
        filters={"season": "2023-24", "our": "TOR", "opp": "BOS", "k": 2},
        columns=["PLAY_TYPE", "PPP_PRED"],
        rows=[
            {"PLAY_TYPE": "Spotup", "PPP_PRED": 1.08, "INTERNAL": "x"},
            {"PLAY_TYPE": "P&R Ball Handler", "PPP_PRED": 0.97, "INTERNAL": "y"},
        ],
        metadata={"source": "synergy"},
    )


def test_csv_and_json_carry_the_same_rows_and_columns():
    contract = ExportContract()

    csv_file = contract.build_export(_payload(), "csv")
    json_file = contract.build_export(_payload(), "json")

    csv_rows = list(csv.DictReader(io.StringIO(csv_file.content.decode("utf-8"))))
    body = json.loads(json_file.content)

    assert list(csv_rows[0]) == body["columns"] == ["PLAY_TYPE", "PPP_PRED"]
    assert [row["PLAY_TYPE"] for row in csv_rows] == [row["PLAY_TYPE"] for row in body["rows"]]
    assert [float(row["PPP_PRED"]) for row in csv_rows] == [row["PPP_PRED"] for row in body["rows"]]
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["filters"]["season"] == "2023-24"


def test_media_type_filename_and_download_header_follow_the_contract():
    file = ExportContract().build_export(_payload(), "CSV")

    assert file.media_type == "text/csv"
    assert file.filename == "baseline-rankings_2023-24_TOR_BOS_2.csv"
    assert file.headers == {
        "Content-Disposition": 'attachment; filename="baseline-rankings_2023-24_TOR_BOS_2.csv"'
    }


def test_pdf_uses_the_renderer_registered_for_the_report():
    contract = ExportContract()
    contract.register_pdf_renderer("baseline_rankings", lambda payload: b"%PDF-" + payload.report.encode())

    file = contract.build_export(_payload(), "pdf")

    assert file.media_type == "application/pdf"
    assert file.content.startswith(b"%PDF-")
    assert file.filename.endswith(".pdf")


def test_pdf_without_renderer_and_unknown_formats_are_rejected():
    contract = ExportContract()

    with pytest.raises(ExportFormatUnavailable):
        contract.build_export(_payload(), "pdf")
    with pytest.raises(ExportFormatUnavailable):
        contract.build_export(_payload(), "xlsx")


def test_rows_missing_a_declared_column_are_rejected():
    payload = ExportPayload(
        report="baseline_rankings", filters={}, columns=["PLAY_TYPE", "PPP_PRED"],
        rows=[{"PLAY_TYPE": "Spotup"}],
    )

    with pytest.raises(ValueError, match="PPP_PRED"):
        ExportContract().build_export(payload, "json")
