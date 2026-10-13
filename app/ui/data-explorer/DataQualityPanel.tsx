"use client";

import React, { useEffect, useState } from "react";
import { fetchDataQualityReport, type DataQualityReport, type QualityCheck, type QualityStatus } from "../../services/dataExplorer";

export type QualityState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; report: DataQualityReport };

const labels: Record<string, string> = {
  schema: "Required columns", completeness: "Missing values",
  duplicates: "Duplicate rows", coverage: "Team and season coverage",
};
const statusLabels: Record<QualityStatus, string> = { pass: "Passed", fail: "Failed", not_run: "Not checked" };
const colors: Record<QualityStatus, string> = { pass: "#86efac", fail: "#fca5a5", not_run: "#fcd34d" };
const list = (value: unknown): string[] => Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
const shortList = (values: string[]) => values.slice(0, 5).join(", ") + (values.length > 5 ? ` and ${values.length - 5} more` : "");

function details(check: QualityCheck): string {
  const d = check.details;
  if (check.status === "not_run") return typeof d.reason === "string" ? d.reason : "This check could not run.";
  if (check.name === "schema") {
    const missing = list(d.missing_columns);
    return missing.length ? `Missing columns: ${shortList(missing)}.` : "All required columns are present.";
  }
  if (check.name === "completeness") {
    const counts = d.null_counts && typeof d.null_counts === "object" ? Object.entries(d.null_counts) : [];
    return counts.length ? `Missing values: ${counts.map(([key, count]) => `${key}: ${count}`).join(", ")}.` : "No missing values in the checked fields.";
  }
  if (check.name === "duplicates") return `${typeof d.duplicate_rows === "number" ? d.duplicate_rows.toLocaleString("en-US") : "Unknown"} duplicate rows.`;
  if (check.name === "coverage") {
    const missing = list(d.missing_season_team_pairs);
    return missing.length ? `Missing team-season pairs: ${shortList(missing)}.`
      : `${d.expected_teams ?? "Unknown"} expected teams covered across ${d.expected_seasons ?? "unknown"} seasons.`;
  }
  return typeof d.reason === "string" ? d.reason : "See the report for this check.";
}

export function DataQualityReportView({ state, onRefresh }: { state: QualityState; onRefresh: () => void }) {
  const report = state.kind === "ready" ? state.report : null;
  return (
    <section className="card" aria-labelledby="quality-heading" aria-busy={state.kind === "loading"} style={{ marginTop: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <h2 id="quality-heading" style={{ margin: 0, fontSize: 18 }}>Data quality</h2>
        <button type="button" className="btn btn--secondary" disabled={state.kind === "loading"} onClick={onRefresh}>
          {state.kind === "error" ? "Retry data checks" : "Refresh data checks"}
        </button>
      </div>
      <div aria-live="polite">
        {state.kind === "loading" && <p role="status">Checking source data…</p>}
        {state.kind === "error" && <div role="alert"><p><strong>Data checks unavailable</strong></p><p>{state.message}</p></div>}
        {report && <>
          <p style={{ color: colors[report.status] }}><strong>
            {report.status === "pass" ? "Listed checks passed" : report.status === "fail" ? "Data checks found problems" : "Some data checks were not run"}
          </strong></p>
          {report.status !== "pass" && <p>Review the results below before relying on this data.</p>}
        </>}
      </div>
      {report && <>
        <p className="muted" style={{ fontSize: 13 }}>
          {report.dataset.name} · {report.dataset.rows.toLocaleString("en-US")} source rows · {report.dataset.seasons.length} seasons · {report.dataset.teams.length} teams
        </p>
        <p className="muted" style={{ fontSize: 13 }}>{report.dataset.scope}. Checked <time dateTime={report.generated_at}>{new Date(report.generated_at).toLocaleString()}</time>.</p>
        <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 10 }}>
          {report.checks.map(check => <li key={check.name} style={{ borderTop: "1px solid rgba(255,255,255,0.12)", paddingTop: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <strong>{labels[check.name] ?? check.name.replaceAll("_", " ")}</strong>
              <strong style={{ color: colors[check.status] }}>{statusLabels[check.status]}</strong>
            </div>
            <p className="muted" style={{ margin: "4px 0", fontSize: 13 }}>{details(check)}</p>
          </li>)}
        </ul>
        <details style={{ fontSize: 12 }}><summary>Dataset version</summary><code style={{ overflowWrap: "anywhere" }}>{report.dataset.version}</code></details>
      </>}
      <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
        These checks cover the play-type source data, including rows outside the preview filters. They do not measure prediction accuracy or validate the separate shot dataset.
      </p>
    </section>
  );
}

export default function DataQualityPanel() {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<QualityState>({ kind: "loading" });
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    setState({ kind: "loading" });
    fetchDataQualityReport(controller.signal).then(report => {
      if (active) setState({ kind: "ready", report });
    }).catch(error => {
      if (active) setState({ kind: "error", message: controller.signal.aborted
        ? "Data checks took too long. Please try again."
        : error instanceof Error ? error.message : "Could not load the data checks. Please try again." });
    }).finally(() => clearTimeout(timer));
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [attempt]);
  return <DataQualityReportView state={state} onRefresh={() => { setState({ kind: "loading" }); setAttempt(n => n + 1); }} />;
}
