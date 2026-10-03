# Unified export contract (SCRUM-484)

Every page that offers a download builds one `ExportPayload` and asks
`ExportContract` for the file. The code lives in
`backend/application/api_coordination/export_contract.py`.

## Payload

| Field | Type | Rule |
|---|---|---|
| `report` | string | Stable report id, e.g. `baseline_rankings`, `context_rankings`, `topk_uplift`, `model_metrics`. |
| `filters` | object | The filters applied on screen, in display order (`season`, `our`, `opp`, `k`, ...). |
| `columns` | string[] | Columns to export, in order. Extra row keys are dropped. |
| `rows` | object[] | Exactly the rows shown on screen. Every row must contain every column. |
| `metadata` | object | Provenance and caveats (source file, metric version, limitations). |

## Formats

| Format | Media type | Content |
|---|---|---|
| `csv` | `text/csv` | Header row from `columns`, one line per row, UTF-8. |
| `json` | `application/json` | `contract_version`, `report`, `filters`, `columns`, `rows`, `metadata`. |
| `pdf` | `application/pdf` | Produced by the renderer registered for the report. |

CSV and JSON always carry the same rows and columns. An unknown format, or
`pdf` for a report with no registered renderer, raises
`ExportFormatUnavailable` (HTTP 400 at the API boundary).

## Response

- `Content-Disposition: attachment; filename="<report>_<filter values>.<format>"`
- File names are deterministic: the report id followed by the filter values in
  order, with unsafe characters replaced by `-`.
- A failed export returns an error the page can show with a retry action; it
  never returns a partial file.

## Existing exports to migrate

| Current endpoint | Report id |
|---|---|
| `/rank-plays/baseline.csv` | `baseline_rankings` |
| `/data/team-playtypes.csv` | `team_playtypes` |
| `/metrics/topk-uplift.csv`, `.json` | `topk_uplift` |
| `/export/playtype-viz.pdf` | `playtype_viz` (PDF renderer) |
| `/export/shotplan.pdf` | `shotplan` (PDF renderer) |

The generic `/export/report` endpoint that serves this contract is SCRUM-485;
wiring the page buttons is SCRUM-486.
