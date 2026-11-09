import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { SavedQuery } from "../app/domain/saved-query";
import type { SavedQueryService } from "../app/services/saved-queries";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));
import SavedQueriesPanel, { errorMessage, openOutcome } from "../app/ui/data-explorer/SavedQueriesPanel";
import { shotFiltersFrom } from "../app/ui/data-explorer/ShotDataExplorer";

const query: SavedQuery = {
  id: "q1",
  name: "Our 5 best play types by PPP this season",
  dataset: "synergy",
  filters: { season: "2024-25", team: "TOR", side: "offense", playType: "", minPoss: 0, search: "" },
  sortKey: "PPP",
  sortDir: "desc",
  rowLimit: 5,
  updatedAt: "2026-11-09T00:00:00Z",
};
const options = { season: ["2023-24", "2024-25"], team: ["BOS", "TOR"], playType: ["Cut", "Spotup"] };
const service: SavedQueryService = {
  list: vi.fn(async () => [query]),
  save: vi.fn(),
  rename: vi.fn(),
  remove: vi.fn(),
};

describe("saved queries panel", () => {
  it("renders the save form and waits for the analyst's list before saving", () => {
    const html = renderToStaticMarkup(
      <SavedQueriesPanel definition={query} options={options} onOpen={() => {}} service={service} />
    );
    expect(html).toContain('aria-label="Saved queries"');
    expect(html).toContain("Question this query answers");
    expect(html).toContain("Loading saved queries");
    expect(html).toMatch(/<button[^>]*type="submit"[^>]*disabled/);
  });

  it("opens a query whose filter values still exist", () => {
    expect(openOutcome(query, options)).toEqual({ load: true, message: `Opened "${query.name}".` });
  });

  it("names a missing season and team instead of loading an empty table", () => {
    const old = { ...query, filters: { ...query.filters, season: "2018-19", team: "SEA" } };
    const outcome = openOutcome(old, options);
    expect(outcome.load).toBe(false);
    expect(outcome.message).toBe("This query uses Season 2018-19 and Team SEA, which are no longer in the data. Change them and save again.");
  });

  it("shows the service's message and falls back when there is none", () => {
    expect(errorMessage(new Error('You already have a query called "X".'), "fallback")).toBe('You already have a query called "X".');
    expect(errorMessage("boom", "The query could not be saved. Try again.")).toBe("The query could not be saved. Try again.");
  });
});

describe("saved shot queries", () => {
  it("restores shot filters and drops unknown keys and result values", () => {
    const shots: SavedQuery = {
      ...query,
      dataset: "shots",
      filters: { season: "2024-25", team: "TOR", opp: "BOS", shotType: "Jump Shot", zone: "Corner 3", result: "made", extra: "x" },
      sortKey: null,
      sortDir: null,
    };
    expect(shotFiltersFrom(shots)).toEqual({ season: "2024-25", team: "TOR", opp: "BOS", shotType: "Jump Shot", zone: "Corner 3", result: "made" });
    expect(shotFiltersFrom({ ...shots, filters: { season: "2024-25", result: "blocked" } }).result).toBe("");
  });
});
