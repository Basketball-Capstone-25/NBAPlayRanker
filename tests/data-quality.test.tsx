import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

import { API_BASE, fetchDataQualityReport } from "../app/infrastructure/api-client";
import { parseDataQualityReport, type DataQualityReport } from "../app/infrastructure/data-quality";
import { DataQualityReportView } from "../app/ui/data-explorer/DataQualityPanel";

function report(): DataQualityReport {
  return {
    status: "pass", safe_for_use: true, generated_at: "2026-10-13T04:00:00Z",
    dataset: { name: "Synergy play-type source", scope: "All source rows; independent of preview filters", rows: 100,
      seasons: ["2024-25"], teams: ["TOR"], version: `frame-sha256:${"a".repeat(64)}` },
    checks: [
      { name: "schema", status: "pass", details: { required_columns: ["PPP"], missing_columns: [] } },
      { name: "completeness", status: "pass", details: { rows: 100, null_counts: {} } },
      { name: "duplicates", status: "pass", details: { key_fields: ["SEASON", "TEAM"], duplicate_rows: 0 } },
      { name: "coverage", status: "pass", details: { expected_seasons: 1, expected_teams: 1, missing_season_team_pairs: [] } },
    ],
  };
}

describe("data-quality contract", () => {
  it("accepts a complete report and preserves the checked source version", () => {
    expect(parseDataQualityReport(report()).dataset.version).toBe(`frame-sha256:${"a".repeat(64)}`);
  });
  it.each([null, {}, { ...report(), checks: [] }, { ...report(), generated_at: "yesterday" },
    { ...report(), dataset: { ...report().dataset, rows: 0 } },
    { ...report(), safe_for_use: false },
    { ...report(), checks: [...report().checks, report().checks[0]] },
    { ...report(), checks: report().checks.slice(0, 3) }])("rejects incomplete or conflicting reports", value => {
    expect(() => parseDataQualityReport(value)).toThrow("incomplete");
  });
  it("rejects a pass label when check details contain missing columns", () => {
    const value = report();
    value.checks[0].details.missing_columns = ["PPP"];
    expect(() => parseDataQualityReport(value)).toThrow("incomplete");
  });
  it("rejects missing check details instead of assuming a pass", () => {
    const value = report();
    value.checks[0].details = {};
    expect(() => parseDataQualityReport(value)).toThrow("incomplete");
  });
  it("rejects empty check configuration disguised as a pass", () => {
    for (const [index, field, value] of [[0, "required_columns", []], [2, "key_fields", []], [3, "expected_teams", 0], [3, "expected_seasons", 0]] as const) {
      const data = report();
      data.checks[index].details[field] = value;
      expect(() => parseDataQualityReport(data)).toThrow("incomplete");
    }
  });
  it("rejects source counts that disagree with completed checks", () => {
    const data = report();
    data.checks[1].details.rows = 0;
    expect(() => parseDataQualityReport(data)).toThrow("incomplete");
    data.checks[1].details.rows = 100;
    data.dataset.teams = [];
    expect(() => parseDataQualityReport(data)).toThrow("incomplete");
  });
  it("accepts a failed report and a check that could not run", () => {
    const value = report();
    value.status = "fail"; value.safe_for_use = false;
    value.checks[0] = { name: "schema", status: "fail", details: { required_columns: ["PPP"], missing_columns: ["PPP"] } };
    value.checks[1] = { name: "completeness", status: "not_run", details: { reason: "missing columns: PPP" } };
    expect(parseDataQualityReport(value).safe_for_use).toBe(false);
  });
});

describe("data-quality requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-session" } } });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("uses analyst session auth without caching or putting tokens in the URL", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchDataQualityReport(signal)).resolves.toEqual(report());
    expect(fetchMock).toHaveBeenCalledWith(`${API_BASE}/data/quality-report`, { cache: "no-store", headers: { Authorization: "Bearer test-session" }, signal });
  });
  it.each([[401, "Sign in again"], [403, "analyst account"], [404, "not available yet"], [503, "unavailable"]])(
    "reports HTTP %s without leaking the server response", async (status, message) => {
      fetchMock.mockResolvedValue(new Response("/private/server/config", { status: Number(status) }));
      await expect(fetchDataQualityReport()).rejects.toThrow(String(message));
      await expect(fetchDataQualityReport()).rejects.not.toThrow("/private/");
    }
  );
  it("never retains a successful report after a refresh fails", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(report())))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }));
    await expect(fetchDataQualityReport()).resolves.toMatchObject({ status: "pass" });
    await expect(fetchDataQualityReport()).rejects.toThrow("unavailable");
  });
  it("handles connection failures", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(fetchDataQualityReport()).rejects.toThrow("Could not reach");
  });
  it("rejects malformed JSON and malformed successful responses", async () => {
    fetchMock.mockResolvedValueOnce(new Response("not-json"))
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "pass" })));
    await expect(fetchDataQualityReport()).rejects.toThrow("unavailable");
    await expect(fetchDataQualityReport()).rejects.toThrow("incomplete");
  });
});

describe("data-quality presentation", () => {
  const render = (state: Parameters<typeof DataQualityReportView>[0]["state"]) =>
    renderToStaticMarkup(<DataQualityReportView state={state} onRefresh={() => {}} />);
  it("shows the dataset scope, four individual checks and version", () => {
    const html = render({ kind: "ready", report: report() });
    expect(html).toContain("Data checks passed");
    expect(html.match(/>Passed</g)).toHaveLength(4);
    expect(html).toContain("independent of preview filters");
    expect(html).toContain("frame-sha256:");
    expect(html).toContain("separate shot dataset");
    expect(html).toContain("<details><summary>View checks</summary>");
    expect(html).not.toContain("<details open");
  });
  it("shows failure details and not-checked results without an overall pass", () => {
    const value = report();
    value.status = "fail"; value.safe_for_use = false;
    value.checks[0] = { name: "schema", status: "fail", details: { missing_columns: ["PPP"] } };
    value.checks[1] = { name: "completeness", status: "not_run", details: { reason: "missing columns: PPP" } };
    const html = render({ kind: "ready", report: value });
    expect(html).toContain("Data checks found problems");
    expect(html).toContain("Missing columns: PPP");
    expect(html).toContain("Not checked");
    expect(html).toContain('<details open=""><summary>View checks</summary>');
    expect(html).not.toContain("Data checks passed");
  });
  it("does not display a previous pass while loading or after an error", () => {
    expect(render({ kind: "loading" })).toContain("Checking source data");
    const html = render({ kind: "error", message: "Please try again." });
    expect(html).toContain("Data checks unavailable");
    expect(html).toContain("Retry data checks");
    expect(html).not.toContain("Passed");
    expect(html).not.toContain("frame-sha256:");
  });
});
