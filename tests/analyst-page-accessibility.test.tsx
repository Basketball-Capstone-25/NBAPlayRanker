import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("../app/services/analytics", () => ({ fetchMlAnalysis: vi.fn(), fetchModelMetrics: vi.fn(), fetchPipelineInfo: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));
import DataExplorer from "../app/ui/data-explorer/DataExplorerClient";
import StatisticalAnalysis from "../app/ui/analytics/_components/StatisticalAnalysisPageClient";
import ShotExplorer from "../app/ui/shot-analysis/_components/ShotExplorerPageClient";
import ShotPlan from "../app/ui/shot-analysis/_components/ShotPlanPageClient";

describe("analyst page controls", () => {
  it("does not report an empty dataset before filters have loaded", () => {
    const html = renderToStaticMarkup(<DataExplorer />);
    expect(html).toContain("Refreshing play-type options");
    expect(html).not.toContain("No recorded play-type rows match");
    expect(html).not.toContain("No rows match these filters");
  });
  it("makes every analysis view reachable with a keyboard button", () => {
    const html = renderToStaticMarkup(<StatisticalAnalysis />);
    for (const label of ["Overview", "Distributions", "Correlations", "Feature Selection", "Model Tuning", "Raw JSON"]) {
      expect(html).toMatch(new RegExp(`<button[^>]*aria-pressed="(?:true|false)"[^>]*>${label}</button>`));
    }
    expect(html).toContain("<summary>Analysis tools</summary>");
    expect(html).not.toContain("<details open");
    expect(html).toContain("Force recompute");
  });

  it.each([ShotExplorer, ShotPlan])("associates shot filters with labels without choosing a matchup", Page => {
    const html = renderToStaticMarkup(<Page />);
    const labels = [...html.matchAll(/<label[^>]*for="([^"]+)"/g)].map(match => match[1]);
    expect(labels.length).toBeGreaterThanOrEqual(6);
    for (const id of labels) expect(html).toMatch(new RegExp(`<(?:select|input) id="${id}"`));
    expect(html).toContain('value="" selected=""');
    expect(html).toContain("Choose a season");
    expect(html).not.toContain('value="TOR" selected');
    expect(html).not.toContain('value="BOS" selected');
  });
});
