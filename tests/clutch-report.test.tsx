import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  clutchFilterProblem,
  clutchQuickCallSituation,
  DEFAULT_CLUTCH,
  describeClutch,
  type ClutchFilters,
  type ClutchReport,
} from "../app/domain/clutch-report";
import { quickCallHref } from "../app/domain/quick-call";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));
import { clutchFilename, clutchReportUrl } from "../app/infrastructure/clutch-report";
import ClutchReportPanel, { ClutchActions, ClutchTable } from "../app/ui/data-explorer/ClutchReportPanel";

const filters: ClutchFilters = { season: "2024-25", team: "TOR", opp: "BOS", ...DEFAULT_CLUTCH };
const split = (shots: number, makes: number, points: number) => ({
  shots, makes, points, points_per_shot: shots ? Math.round((points / shots) * 1000) / 1000 : null, small_sample: shots < 20,
});
const report: ClutchReport = {
  filters: { season: "2024-25", team: "TOR", opp: "BOS", period_from: 4, seconds_left: 300, margin_min: -5, margin_max: 5 },
  small_sample_threshold: 20,
  totals: { team: split(22, 11, 32), opp: split(1, 1, 2) },
  rows: [
    { zone: "rim", label: "At the rim", team: split(2, 1, 2), opp: split(1, 1, 2) },
    { zone: "corner3", label: "Corner 3", team: split(20, 10, 30), opp: split(0, 0, 0) },
  ],
  message: null,
  dataset_version: "sha256:test",
};

describe("clutch report filters (SCRUM-629)", () => {
  it("accepts the default late-game filters", () => {
    expect(clutchFilterProblem(filters)).toBe("");
    expect(describeClutch(filters)).toBe("4th quarter and overtime, 5:00 or less left, margin -5 to +5");
  });

  it("explains each filter it cannot use, matching the API limits", () => {
    expect(clutchFilterProblem({ ...filters, opp: "" })).toBe("Pick your team and the opponent.");
    expect(clutchFilterProblem({ ...filters, opp: "tor" })).toBe("Pick two different teams.");
    expect(clutchFilterProblem({ ...filters, periodFrom: 5 })).toMatch(/^Start period/);
    expect(clutchFilterProblem({ ...filters, secondsLeft: 721 })).toBe("Time left must be between 1 second and 12 minutes.");
    expect(clutchFilterProblem({ ...filters, marginMin: NaN })).toMatch(/^Score margin/);
    expect(clutchFilterProblem({ ...filters, marginMin: 3, marginMax: -3 })).toBe("The lowest margin must not be above the highest margin.");
  });

  it("builds API and download addresses from the filters", () => {
    expect(clutchReportUrl(filters)).toContain(
      "/data/clutch-report?season=2024-25&team=TOR&opp=BOS&period_from=4&seconds_left=300&margin_min=-5&margin_max=5"
    );
    expect(clutchReportUrl(filters, "pdf")).toContain("/data/clutch-report.pdf?");
    expect(clutchFilename(filters, "csv")).toBe("clutch_TOR_vs_BOS_2024-25.csv");
  });
});

describe("clutch report as a Quick-Call preset", () => {
  it("uses the matchup, the time limit and the margin closest to a tie", () => {
    expect(quickCallHref(clutchQuickCallSituation(filters))).toBe(
      "/context/quick-call?season=2024-25&our=TOR&opp=BOS&margin=0&period=4&time=300"
    );
    expect(clutchQuickCallSituation({ ...filters, marginMin: -5, marginMax: -1 }).margin).toBe(-1);
    expect(clutchQuickCallSituation({ ...filters, marginMin: 2, marginMax: 6 }).margin).toBe(2);
    expect(clutchQuickCallSituation({ ...filters, periodFrom: 2 }).period).toBe(4);
  });
});

describe("clutch report display", () => {
  it("shows both teams side by side by zone with totals", () => {
    const html = renderToStaticMarkup(<ClutchTable report={report} />);
    for (const text of ["TOR", "BOS", "At the rim", "Corner 3", "All zones", "1.50", "Pts/shot", "Data version sha256:test"])
      expect(html).toContain(text);
    expect(html.indexOf("At the rim")).toBeLessThan(html.indexOf("Corner 3"));
  });

  it("flags zones with fewer than 20 shots and not a zone with exactly 20", () => {
    const html = renderToStaticMarkup(<ClutchTable report={report} />);
    expect(html.match(/data-small-sample="true"/g)).toHaveLength(4); // rim x2, BOS corner 3, BOS total
    expect(html).toContain("Fewer than 20 shots: a small sample");
    expect(html).toMatch(/<td[^>]*>20<\/td>/);
  });

  it("explains an empty report instead of an empty table", () => {
    const empty = { ...report, rows: [], message: "No shots match these situations. Try a wider margin." };
    const html = renderToStaticMarkup(<ClutchTable report={empty} />);
    expect(html).toContain("Try a wider margin.");
    expect(html).not.toContain("<table");
  });

  it("offers PDF, CSV and a Quick-Call preset link for the report on screen", () => {
    const html = renderToStaticMarkup(<ClutchActions filters={filters} downloading="" onDownload={() => {}} />);
    expect(html).toContain("Download PDF");
    expect(html).toContain("Download CSV");
    expect(html).toContain('href="/context/quick-call?season=2024-25&amp;our=TOR&amp;opp=BOS&amp;margin=0&amp;period=4&amp;time=300"');
  });

  it("renders the filter form before a report is built", () => {
    const html = renderToStaticMarkup(<ClutchReportPanel seasons={["2024-25"]} teams={["BOS", "TOR"]} season="2024-25" team="TOR" />);
    for (const text of ['aria-label="Clutch report"', "Your team", "Opponent", "Time left in period", "Margin from", "Build report"])
      expect(html).toContain(text);
    expect(html).not.toContain("<table");
  });
});
