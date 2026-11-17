import { describe, expect, it } from "vitest";
import { parseTeamComparison, teamComparisonKey, type TeamComparison } from "../app/domain/team-comparison";

const filters = { season: "2024-25", team: "TOR", opponent: "BOS" };
const report = (): TeamComparison => ({
  status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "synergy_regular_season_v2",
  team_possessions: 100, opponent_possessions: 200,
  rows: [
    { play_type: "Cut", team: { possessions: 25, points: 30, usage_pct: 25, ppp: 1.2 },
      opponent: { possessions: 100, points: 100, usage_pct: 50, ppp: 1 }, difference: { usage_pp: -25, ppp: 0.2 } },
    { play_type: "Isolation", team: { possessions: 75, points: 75, usage_pct: 75, ppp: 1 },
      opponent: null, difference: { usage_pp: null, ppp: null } },
    { play_type: "Transition", team: null, opponent: { possessions: 100, points: 150, usage_pct: 50, ppp: 1.5 },
      difference: { usage_pp: null, ppp: null } },
  ],
});

describe("two-team profile contract", () => {
  it("keeps unmatched categories and independent full-team denominators", () => {
    expect(parseTeamComparison(report(), filters)).toEqual(report());
    expect(teamComparisonKey(filters)).not.toBe(teamComparisonKey({ ...filters, team: "BOS", opponent: "TOR" }));
  });
  it("keeps the available profile when the other team has no season data", () => {
    const value = report(); value.status = "no_data"; value.opponent_possessions = null;
    value.rows = value.rows.filter(row => row.team).map(row => ({ ...row, opponent: null, difference: { usage_pp: null, ppp: null } }));
    expect(parseTeamComparison(value, filters).rows).toHaveLength(2);
    value.team_possessions = null; value.rows = [];
    expect(parseTeamComparison(value, filters)).toEqual(value);
  });
  it("retains zero counts without inventing rates", () => {
    const value = report(); value.team_possessions = 0;
    for (const row of value.rows) {
      if (row.team) row.team = { possessions: 0, points: 0, usage_pct: null, ppp: null };
      row.difference = { usage_pp: null, ppp: null };
    }
    expect(parseTeamComparison(value, filters)).toEqual(value);
  });
  it.each([
    ["wrong team", (v: any) => { v.filters.team = "LAL"; }],
    ["wrong opponent", (v: any) => { v.filters.opponent = "LAL"; }],
    ["wrong season", (v: any) => { v.filters.season = "2023-24"; }],
    ["defense", (v: any) => { v.filters.side = "defense"; }],
    ["playoffs", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["missing source", (v: any) => { v.dataset_version = ""; }],
    ["missing aggregation", (v: any) => { v.aggregation_version = null; }],
    ["reused denominator", (v: any) => { v.rows[0].team.usage_pct = 12.5; }],
    ["wrong PPP", (v: any) => { v.rows[0].team.ppp = 1.1; }],
    ["relative percentage difference", (v: any) => { v.rows[0].difference.usage_pp = -50; }],
    ["invented missing difference", (v: any) => { v.rows[1].difference.ppp = 0; }],
    ["fractional count", (v: any) => { v.rows[0].team.possessions = 25.5; }],
    ["negative count", (v: any) => { v.rows[0].team.points = -1; }],
    ["unsafe count", (v: any) => { v.team_possessions = Number.MAX_SAFE_INTEGER + 1; }],
    ["nonfinite rate", (v: any) => { v.rows[0].team.ppp = Infinity; }],
    ["missing rate", (v: any) => { v.rows[0].team.usage_pct = null; }],
    ["lost category", (v: any) => { v.rows.pop(); }],
    ["duplicate category", (v: any) => { v.rows.push(v.rows[0]); }],
    ["empty category", (v: any) => { v.rows[0].play_type = " "; }],
    ["empty row", (v: any) => { v.rows[0].team = v.rows[0].opponent = null; }],
    ["missing total", (v: any) => { v.team_possessions = null; }],
    ["wrong status", (v: any) => { v.status = "no_data"; }],
    ["empty success", (v: any) => { v.rows = []; }],
  ])("rejects %s", (_name, mutate) => {
    const value = report(); (mutate as (v: any) => void)(value);
    expect(() => parseTeamComparison(value, filters)).toThrow("incomplete");
  });
  it("rejects identical teams", () => {
    const value = report(); value.filters.opponent = "TOR";
    expect(() => parseTeamComparison(value, { ...filters, opponent: "TOR" })).toThrow("incomplete");
  });
});
