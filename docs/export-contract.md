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
- Negative numeric filters keep their minus sign. A context margin of `-2`
  therefore produces `..._-2...`, while a margin of `2` produces `..._2...`,
  in CSV, JSON and PDF filenames. Text sanitization and the report prefix stay
  the same; quotes, path separators and line breaks cannot enter the header.
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

## Export the displayed result (SCRUM-486)

Matchup, Context and Model Metrics share one format selector and Export button.
The control is disabled while results are loading or no longer match the
selected settings. CSV, JSON and PDF use the same saved result. Context exports
include the adjustment shown in its table; exporting does not train a model or
run the ranking again.

`POST /export/report` accepts `{format, payload}` for `baseline_rankings`,
`context_rankings` and `model_metrics`. The first two require a coach account;
metrics require an analyst account. The server sets
`metadata.export_source = displayed_result_snapshot`, so a submitted snapshot
is not presented as a fresh server calculation.

The request is limited to 256 KiB, 200 rows and 16 distinct columns. Rows must
contain exactly those columns, using finite numbers, booleans, null or short
text. Metadata is bounded separately. Invalid requests return an error without
an attachment. CSV escapes formula-like text for spreadsheet safety; numeric
cells retain their values. PDF wraps long cells, repeats headers and splits
wide tables into readable column groups. JSON and PDF retain filters and report
details; CSV retains the table, with the report and settings in the filename.

The browser posts to the same-origin `/api/exports/report` route. It verifies
the session with Supabase, then forwards its bearer token to the fixed backend
path. A JSON preflight checks that the file can be generated. A form POST starts
the browser's regular attachment download. No access token is put in a URL.
Responses are private and are not cached. A failed preflight shows Retry export;
the page says Download requested after submission, because saving the file is
handled by the browser.
