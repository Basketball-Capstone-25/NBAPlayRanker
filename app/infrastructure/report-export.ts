export type ExportFormat = "csv" | "json" | "pdf";
export type ExportScalar = string | number | boolean | null;
export type ReportPayload = {
  report: "baseline_rankings" | "context_rankings" | "model_metrics";
  filters: Record<string, ExportScalar>;
  columns: string[];
  rows: Record<string, ExportScalar>[];
  metadata: Record<string, unknown>;
};

export const REPORT_EXPORT_PATH = "/api/exports/report";

export function resultKey(filters: Record<string, ExportScalar>): string {
  return JSON.stringify(filters);
}

const numberOrNull = (value: number): number | null => Number.isFinite(value) ? value : null;

export function baselineReport(
  filters: ReportPayload["filters"],
  rows: { playType: string; pppPred: number; pppOff: number; pppDef: number; pppGap: number }[],
): ReportPayload {
  return {
    report: "baseline_rankings", filters: { ...filters },
    columns: ["Play type", "Predicted PPP", "Our PPP", "Opponent PPP", "PPP gap"],
    rows: rows.map(row => ({
      "Play type": row.playType, "Predicted PPP": numberOrNull(row.pppPred),
      "Our PPP": numberOrNull(row.pppOff), "Opponent PPP": numberOrNull(row.pppDef),
      "PPP gap": numberOrNull(row.pppGap),
    })),
    metadata: { title: "Matchup rankings", source: "displayed result snapshot" },
  };
}

export function contextReport(
  filters: ReportPayload["filters"],
  rows: { playType: string; finalPPP_display: number; mlPPP: number; baselinePPP: number;
    deltaPPP_display: number; overlay: { overlayPPP: number } }[],
): ReportPayload {
  return {
    report: "context_rankings", filters: { ...filters },
    columns: ["Play type", "Final PPP", "ML PPP", "Baseline PPP", "Change in PPP", "Overlay PPP"],
    rows: rows.map(row => ({
      "Play type": row.playType, "Final PPP": numberOrNull(row.finalPPP_display),
      "ML PPP": numberOrNull(row.mlPPP), "Baseline PPP": numberOrNull(row.baselinePPP),
      "Change in PPP": numberOrNull(row.deltaPPP_display), "Overlay PPP": numberOrNull(row.overlay.overlayPPP),
    })),
    metadata: { title: "Context rankings", source: "displayed result snapshot",
      limitations: "Final PPP includes the selected context overlay. These are modeled rankings, not observed outcomes." },
  };
}

export function metricsReport(data: {
  n_splits: number;
  metrics: { model: string; RMSE_mean: number; RMSE_std: number; MAE_mean: number; MAE_std: number; R2_mean: number; R2_std: number }[];
  rf_vs_baseline_t: number | null;
  rf_vs_baseline_p: number | null;
}): ReportPayload {
  return {
    report: "model_metrics", filters: { n_splits: data.n_splits },
    columns: ["Model", "RMSE mean", "RMSE SD", "MAE mean", "MAE SD", "R2 mean", "R2 SD"],
    rows: data.metrics.map(row => ({
      Model: row.model, "RMSE mean": numberOrNull(row.RMSE_mean), "RMSE SD": numberOrNull(row.RMSE_std),
      "MAE mean": numberOrNull(row.MAE_mean), "MAE SD": numberOrNull(row.MAE_std),
      "R2 mean": numberOrNull(row.R2_mean), "R2 SD": numberOrNull(row.R2_std),
    })),
    metadata: { title: "Holdout model metrics", source: "displayed result snapshot",
      rf_vs_baseline_t: data.rf_vs_baseline_t, rf_vs_baseline_p: data.rf_vs_baseline_p,
      limitations: "Retrospective holdout evaluation; this table does not establish pre-game forecast accuracy." },
  };
}

/** Check the real response before starting a native HTTP attachment download. */
export async function prepareReportExport(payload: ReportPayload, format: ExportFormat, signal?: AbortSignal): Promise<string> {
  const body = JSON.stringify({ format, payload });
  const response = await fetch(REPORT_EXPORT_PATH, {
    method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json" }, body, signal,
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || result?.ready !== true) {
    throw new Error(typeof result?.error === "string" ? result.error : "Could not prepare this export. Please try again.");
  }
  return body;
}

export function submitReportExport(body: string): void {
  const form = document.createElement("form");
  const field = document.createElement("input");
  form.method = "POST";
  form.action = REPORT_EXPORT_PATH;
  form.target = "_blank";
  form.rel = "noopener";
  form.hidden = true;
  field.type = "hidden";
  field.name = "request";
  field.value = body;
  form.appendChild(field);
  document.body.appendChild(form);
  try { form.submit(); } finally { form.remove(); }
}
