import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  query: { result: null as any, inputs: null as any, changed: false, loading: false, error: null as string | null, run: vi.fn(), reset: vi.fn() },
  pdf: vi.fn(() => "/saved-shot-plan.pdf"), csv: vi.fn(() => "/saved-shots.csv"),
}));
vi.mock("../app/ui/shot-analysis/_components/useShotResult", () => ({ useShotResult: () => mocks.query }));
vi.mock("../app/services/shotAnalysis", () => ({
  fetchMetaOptions: vi.fn(), fetchPbpMetaOptions: vi.fn(), fetchShotHeatmap: vi.fn(), fetchShotPlanRank: vi.fn(), fetchPbpShotsPreview: vi.fn(),
  getShotPlanPdfUrl: mocks.pdf, getPbpShotsCsvUrl: mocks.csv, authenticatedDownload: vi.fn(),
}));
import ShotPlan from "../app/ui/shot-analysis/_components/ShotPlanPageClient";
import ShotHeatmap from "../app/ui/shot-analysis/_components/ShotHeatmapPageClient";
import ShotExplorer from "../app/ui/shot-analysis/_components/ShotExplorerPageClient";
import { ShotHeatmapCounts } from "../app/ui/shot-analysis/_components/ShotHeatmapCounts";

const saved = { season: "2024-25", team: "LAL", our: "LAL", opp: "DEN", shotType: "Layup", zone: "rim", maxShots: 1000, k: 3, wOff: .4, limit: 20 };
beforeEach(() => {
  Object.assign(mocks.query, { result: null, inputs: null, changed: false, loading: false, error: null });
  vi.clearAllMocks();
});

describe("shot page result and export guards", () => {
  it.each(["changed", "loading", "error"])("disables export and removes old results while %s", status => {
    if (status === "error") mocks.query.error = "Request failed";
    else mocks.query[status] = true;
    const plan = renderToStaticMarkup(<ShotPlan />), explorer = renderToStaticMarkup(<ShotExplorer />), heatmap = renderToStaticMarkup(<ShotHeatmap />);
    expect(plan).toMatch(/<button[^>]*disabled=""[^>]*>Export PDF/);
    expect(explorer).toMatch(/<button[^>]*disabled=""[^>]*>Export CSV/);
    expect(heatmap).not.toContain("Download PNG");
    expect(heatmap).not.toContain("Copy summary");
    expect(plan).not.toContain("Top Shot Types");
    expect(mocks.pdf).not.toHaveBeenCalled(); expect(mocks.csv).not.toHaveBeenCalled();
    if (status === "changed") for (const page of [plan, explorer, heatmap]) expect(page).toContain("changed.");
  });

  it("uses the successful shot-plan inputs for the PDF and names the scoring unit", () => {
    mocks.query.inputs = saved;
    mocks.query.result = { rank: { season: saved.season, our_team: saved.our, opp_team: saved.opp, k: saved.k, w_off: saved.wOff, w_def: .6,
      top_shot_types: [{ SHOT_TYPE: "Layup", EPA_PRED: 1.2, EPA_OFF_SHRUNK: null, EPA_DEF_SHRUNK: 1.1, attempts_OFF: 100, attempts_DEF: 80 }], top_zones: [] }, heatmap: { image_base64: "test", caption: "Saved matchup" } };
    const html = renderToStaticMarkup(<ShotPlan />);
    expect(mocks.pdf).toHaveBeenCalledWith(saved);
    expect(html).toContain("Estimated points/attempt");
    expect(html).toContain("Unavailable");
    expect(html).toContain("season-long offense");
    expect(html).not.toContain("opponent-specific weaknesses");
  });

  it("labels and names a heatmap download using its successful request", () => {
    mocks.query.inputs = saved; mocks.query.result = { image_base64: "test", caption: "Saved matchup", n_shots: 1000, dataset_version: `sha256:${"a".repeat(64)}` };
    const html = renderToStaticMarkup(<ShotHeatmap />);
    expect(html).toContain("LAL vs DEN"); expect(html).toContain("Season: 2024-25");
    expect(html).toContain('href="/api/exports/shot-heatmap?season=2024-25&amp;team=LAL&amp;opp=DEN&amp;max_shots=1000&amp;version=sha256%3A' + "a".repeat(64) + '&amp;shot_type=Layup&amp;zone=rim"');
    expect(html).toContain('target="_blank"'); expect(html).not.toContain('href="data:'); expect(html).not.toContain('download=');
    expect(html).toContain("Attempt"); expect(html).toContain("not shooting percentage");
  });

  it("requires another render when the displayed source version is missing", () => {
    mocks.query.inputs = saved; mocks.query.result = { image_base64: "test", n_shots: 1000 };
    const html = renderToStaticMarkup(<ShotHeatmap />);
    expect(html).not.toContain("Download PNG");
    expect(html).toContain("Render again to enable PNG download.");
  });

  it("exports the successful explorer filters and explains an empty result", () => {
    mocks.query.inputs = saved;
    mocks.query.result = { season: saved.season, team: saved.team, opp: saved.opp, shot_type: saved.shotType, zone: saved.zone, returned_rows: 0, total_rows: 0, columns: [], rows: [] };
    const html = renderToStaticMarkup(<ShotExplorer />);
    expect(mocks.csv).toHaveBeenCalledWith({ season: "2024-25", team: "LAL", opp: "DEN", shotType: "Layup", zone: "rim", limit: 5000 });
    expect(html).toContain("No shot attempts match these filters");
    const preview = html.split("<h2>Preview</h2>")[1];
    expect(preview).toContain("2024-25"); expect(preview).toContain("LAL"); expect(preview).not.toContain("TOR");
  });

  it("separates matched attempts from plotted attempts and explains each exclusion", () => {
    const html = renderToStaticMarkup(<ShotHeatmapCounts heatmap={{ image_base64: "image", n_shots_total: 1300, n_shots_rendered: 1000, n_shots_excluded_location: 20, n_shots_outside_halfcourt: 30, n_shots_omitted_by_cap: 250 }} />);
    for (const text of ["Matching attempts: 1,300", "Attempts shown: 1,000", "Unusable locations excluded: 20", "Outside the half court: 30", "Not shown due to sample cap: 250"]) expect(html).toContain(text);
    expect(renderToStaticMarkup(<ShotHeatmapCounts heatmap={{ image_base64: "image" }} />)).toBe("");
  });
});
