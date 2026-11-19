import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { refreshedShotFilters } from "../app/ui/data-explorer/ShotDataExplorer";
import AnalystSeasonSummary from "../app/ui/data-explorer/AnalystSeasonSummary";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));

const options = { seasons: ["2023-24", "2024-25"], teams: ["TOR", "BOS"], shotTypes: ["2PT"], zones: ["Restricted Area"] };
const filters = { season: "2024-25", team: "TOR", opp: "BOS", shotType: "2PT", zone: "Restricted Area", result: "made" as const };

describe("filters after a dataset reload", () => {
  it("retains valid searched shot filters in a new snapshot", () => {
    const refreshed = refreshedShotFilters(filters, options);
    expect(refreshed).toEqual(filters);
    expect(refreshed).not.toBe(filters);
  });
  it("clears removed options without selecting a different team or season", () => {
    expect(refreshedShotFilters(filters, { seasons: ["2022-23"], teams: ["LAL"], shotTypes: [], zones: [] }))
      .toEqual({ season: "", team: "", opp: "", shotType: "", zone: "", result: "made" });
  });
  it("retains valid required filters but clears a self-opponent and missing optional choices", () => {
    expect(refreshedShotFilters({ ...filters, opp: "TOR", shotType: "removed", zone: "removed" }, options))
      .toEqual({ ...filters, opp: "", shotType: "", zone: "" });
  });
  it("keeps all-opponent, all-zone and all-result selections", () => {
    const all = { ...filters, opp: "", shotType: "", zone: "", result: "" as const };
    expect(refreshedShotFilters(all, options)).toEqual(all);
  });
  it("does not request a summary against old tables while play-type metadata refreshes", () => {
    const html = renderToStaticMarkup(<AnalystSeasonSummary season="2024-25" team="TOR" side="offense"
      teams={["TOR", "BOS"]} refreshing refreshVersion={1} />);
    expect(html).toContain("Refresh the play-type view before loading its summary");
    expect(html).not.toContain("Loading season summary");
    expect(html).not.toContain("Observed team PPP");
  });
});
