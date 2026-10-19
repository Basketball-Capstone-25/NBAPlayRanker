import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { baselineReport, contextReport, metricsReport, prepareReportExport, resultKey, submitReportExport } from "../app/infrastructure/report-export";
import { ReportExportView } from "../app/ui/_shared/components/ReportExport";

const filters = { season: "2024-25", our: "TOR", opp: "BOS", k: 3, w_off: 0.7, w_def: 0.3 };
const baseline = () => baselineReport(filters, [
  { playType: "Cut", pppPred: 1.2, pppOff: 1.3, pppDef: 1.1, pppGap: 0.2 },
  { playType: "Spotup", pppPred: 1.1, pppOff: 1.2, pppDef: 1, pppGap: 0.2 },
]);

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("displayed report exports", () => {
  it("keeps the displayed baseline order, numbers and normalized weights", () => {
    const report = baseline();
    expect(report.filters).toEqual(filters);
    expect(report.rows.map(row => row["Play type"])).toEqual(["Cut", "Spotup"]);
    expect(report.rows[0]["Predicted PPP"]).toBe(1.2);
    expect(Object.keys(report.rows[0])).toEqual(report.columns);
    report.filters.our = "LAL";
    expect(filters.our).toBe("TOR");
  });

  it("uses the displayed context overlay and ordering instead of unadjusted API values", () => {
    const report = contextReport({ ...filters, margin: -2, period: 4, time_remaining: 120, overlay: true, overlay_strength: 1.5 }, [
      { playType: "Cut", finalPPP_display: 1.25, mlPPP: 1.2, baselinePPP: 1.1, deltaPPP_display: 0.15, overlay: { overlayPPP: 0.05 } },
      { playType: "Spotup", finalPPP_display: 1.22, mlPPP: 1.21, baselinePPP: 1.15, deltaPPP_display: 0.07, overlay: { overlayPPP: 0.01 } },
    ]);
    expect(report.rows[0]).toMatchObject({ "Play type": "Cut", "Final PPP": 1.25, "Overlay PPP": 0.05, "Change in PPP": 0.15 });
    expect(report.filters).toMatchObject({ margin: -2, period: 4, time_remaining: 120, overlay: true, overlay_strength: 1.5 });
    expect(report.rows.map(row => row["Play type"])).toEqual(["Cut", "Spotup"]);
  });

  it("exports all loaded holdout models and the response's season folds", () => {
    const report = metricsReport({ n_splits: 5, rf_vs_baseline_t: 2, rf_vs_baseline_p: 0.04,
      metrics: [{ model: "Ridge", RMSE_mean: 0.06, RMSE_std: 0.01, MAE_mean: 0.05, MAE_std: 0.02, R2_mean: 0.8, R2_std: 0.1 }] });
    expect(report.filters).toEqual({ n_splits: 5 });
    expect(report.rows[0]).toMatchObject({ Model: "Ridge", "RMSE mean": 0.06, "MAE mean": 0.05 });
    expect(report.metadata.rf_vs_baseline_p).toBe(0.04);
    expect(report.metadata.limitations).toContain("Retrospective");
  });

  it("represents missing numbers as empty values, never a made-up zero", () => {
    const report = baselineReport(filters, [{ playType: "Cut", pppPred: NaN, pppOff: Infinity, pppDef: 1.1, pppGap: NaN }]);
    expect(report.rows[0]).toMatchObject({ "Predicted PPP": null, "Our PPP": null, "PPP gap": null });
  });

  it("distinguishes a new filter selection from the successfully loaded result", () => {
    expect(resultKey(filters)).not.toBe(resultKey({ ...filters, our: "LAL" }));
    expect(resultKey(filters)).not.toBe(resultKey({ ...filters, w_off: 0.5 }));
    expect(resultKey({ margin: -2, period: 4 })).not.toBe(resultKey({ margin: 2, period: 4 }));
  });
});

describe("report export downloads", () => {
  it("prepares the exact snapshot using cookies without credentials or filters in the URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ ready: true }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    const body = await prepareReportExport(baseline(), "pdf", signal);
    expect(fetchMock).toHaveBeenCalledWith("/api/exports/report", { method: "POST", credentials: "same-origin", cache: "no-store", headers: { "Content-Type": "application/json" }, body, signal });
    expect(JSON.parse(body)).toEqual({ format: "pdf", payload: baseline() });
  });

  it.each([401, 403, 422, 502])("shows a preparation error for HTTP %s", async status => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ error: "Please run this report again." }, { status })));
    await expect(prepareReportExport(baseline(), "csv")).rejects.toThrow("run this report again");
  });

  it("does not treat an invalid successful response as a prepared file", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ success: true })));
    await expect(prepareReportExport(baseline(), "csv")).rejects.toThrow("Could not prepare");
  });

  it("uses a native form attachment and cleans up without creating a Blob URL", () => {
    const form = { method: "", action: "", target: "", rel: "", hidden: false, appendChild: vi.fn(), submit: vi.fn(), remove: vi.fn() };
    const field = { type: "", name: "", value: "" };
    const appendChild = vi.fn();
    vi.stubGlobal("document", { createElement: vi.fn(kind => kind === "form" ? form : field), body: { appendChild } });
    const blob = vi.spyOn(URL, "createObjectURL");
    const body = JSON.stringify({ format: "csv", payload: baseline() });
    submitReportExport(body);
    expect(form).toMatchObject({ method: "POST", action: "/api/exports/report", target: "_blank", rel: "noopener", hidden: true });
    expect(field).toEqual({ type: "hidden", name: "request", value: body });
    expect(form.submit).toHaveBeenCalledOnce();
    expect(form.remove).toHaveBeenCalledOnce();
    expect(blob).not.toHaveBeenCalled();
  });
});

describe("export control", () => {
  const render = (state: Parameters<typeof ReportExportView>[0]["state"], disabled?: string) => renderToStaticMarkup(
    <ReportExportView format="csv" onFormat={() => {}} onExport={() => {}} state={state} disabled={disabled} />,
  );
  it("offers the same three formats and disables export while a report is stale", () => {
    const html = render({ kind: "idle" }, "Inputs changed. Run the report again.");
    expect(html).toContain('value="csv"'); expect(html).toContain('value="pdf"'); expect(html).toContain('value="json"');
    expect(html).toContain("disabled"); expect(html).toContain("Inputs changed");
  });
  it("shows preparation and retry states accessibly", () => {
    expect(render({ kind: "preparing" })).toContain("Preparing your file");
    const html = render({ kind: "error", message: "Sign in again." });
    expect(html).toContain('role="alert"'); expect(html).toContain("Sign in again."); expect(html).toContain("Retry export");
  });
  it("says download requested without claiming the file was saved", () => {
    const html = render({ kind: "requested", message: "Download requested. Check your browser’s Downloads folder." });
    expect(html).toContain("Download requested"); expect(html).not.toContain("saved");
  });
});
