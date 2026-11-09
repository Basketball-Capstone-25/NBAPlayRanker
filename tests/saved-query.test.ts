import { describe, expect, it } from "vitest";
import {
  describeMissing,
  missingValues,
  normalizeQueryName,
  validateName,
  type QueryDefinition,
  type SavedQuery,
} from "../app/domain/saved-query";

const definition: QueryDefinition = {
  dataset: "synergy",
  filters: { season: "2024-25", team: "TOR", side: "offense", playType: "", minPoss: 20, search: "" },
  sortKey: "PPP",
  sortDir: "desc",
  rowLimit: 5,
};
const saved = (id: string, name: string): SavedQuery => ({ ...definition, id, name, updatedAt: "2026-11-09T00:00:00Z" });

describe("saved query names (SCRUM-771)", () => {
  it("accepts a new question as the name", () => {
    expect(validateName("Our 5 best play types by PPP this season", [])).toBe("");
  });

  it("rejects an empty or blank name", () => {
    expect(validateName("", [])).toMatch(/Give the query a name/);
    expect(validateName("   ", [])).toMatch(/Give the query a name/);
  });

  it("rejects a name the analyst already used, ignoring case and extra spaces", () => {
    const existing = [saved("a", "Our 5 best play types by PPP this season")];
    expect(validateName("  our 5 BEST play types by ppp   this season ", existing)).toMatch(/already have a query/);
  });

  it("lets a query keep its own name when it is renamed", () => {
    const existing = [saved("a", "Top 5 by PPP")];
    expect(validateName("top 5 by ppp", existing, "a")).toBe("");
  });

  it("rejects names over 120 characters", () => {
    expect(validateName("x".repeat(121), [])).toMatch(/120 characters/);
    expect(validateName("x".repeat(120), [])).toBe("");
  });

  it("tidies spaces the same way the table does", () => {
    expect(normalizeQueryName("  Top   5  ")).toBe("Top 5");
  });
});

describe("saved filter values that are no longer in the data (SCRUM-771)", () => {
  const options = { season: ["2023-24", "2024-25"], team: ["BOS", "TOR"], playType: ["Isolation", "Spotup"] };

  it("finds nothing missing when every value still exists", () => {
    expect(missingValues(definition, options)).toEqual([]);
  });

  it("names a removed season and team instead of returning an empty result", () => {
    const old = { ...definition, filters: { ...definition.filters, season: "2018-19", team: "SEA" } };
    const missing = missingValues(old, options);
    expect(missing).toEqual(["Season 2018-19", "Team SEA"]);
    expect(describeMissing(missing)).toBe("This query uses Season 2018-19 and Team SEA, which are no longer in the data. Change them and save again.");
  });

  it("treats blank values as all and skips filters with no option list", () => {
    const blank = { ...definition, filters: { season: "2024-25", playType: "", minPoss: 20, search: "spot" } };
    expect(missingValues(blank, options)).toEqual([]);
  });
});
