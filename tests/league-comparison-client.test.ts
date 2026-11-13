import { describe, expect, it } from "vitest";
import { leagueKey, parseLeagueComparison, type LeagueComparison, type LeagueFilters } from "../app/domain/league-comparison";

const filters: LeagueFilters = { season: "2024-25", team: "TOR", side: "offense" };
const report = (): LeagueComparison => ({
  status: "ok", filters: { ...filters, season_phase: "regular_season" }, selected_team_in_league: true,
  dataset_version: "fixture-v1", aggregation_version: "synergy_regular_season_v2",
  rows: [{ play_type: "Cut",
    team: { possessions: 100, field_goal_attempts: 80, fg_pct: 0.5, efg_pct: 0.6, tov_pct: 0.2, ft_pct: 0.1 },
    league: { possessions: 1000, field_goal_attempts: 800, fg_pct: 0.6, efg_pct: 0.6, tov_pct: 0.1, ft_pct: 0.2 },
    difference_pp: { fg_pct: -10, efg_pct: 0, tov_pct: 10, ft_pct: -10 },
  }],
});

describe("league comparison contract", () => {
  it("preserves signed percentage-point differences and the one-team scope", () => {
    expect(parseLeagueComparison(report(), filters)).toEqual(report());
    expect(leagueKey(filters)).not.toBe(leagueKey({ ...filters, side: "defense" }));
  });
  it("accepts eFG above 100% and keeps unavailable attempts separate from possessions", () => {
    const value = report(), row = value.rows[0];
    row.team!.efg_pct = 1.2; row.difference_pp.efg_pct = 60;
    expect(parseLeagueComparison(value, filters)).toEqual(value);
    row.team!.field_goal_attempts = 0; row.team!.fg_pct = row.team!.efg_pct = null;
    row.difference_pp.fg_pct = row.difference_pp.efg_pct = null;
    expect(parseLeagueComparison(value, filters).rows[0].team!.tov_pct).toBe(0.2);
  });
  it("keeps missing league categories and empty selections unavailable", () => {
    const value = report(); value.rows[0].league = null;
    value.rows[0].difference_pp = { fg_pct: null, efg_pct: null, tov_pct: null, ft_pct: null };
    expect(parseLeagueComparison(value, filters)).toEqual(value);
    value.status = "no_data"; value.rows = [];
    expect(parseLeagueComparison(value, filters).rows).toEqual([]);
  });
  it.each(["season", "team", "side"] as const)("rejects a different %s", field => {
    expect(() => parseLeagueComparison(report(), { ...filters, [field]: "different" } as LeagueFilters)).toThrow("incomplete");
  });
  it.each([
    ["playoffs", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["missing source", (v: any) => { v.dataset_version = ""; }],
    ["different benchmark", (v: any) => { v.selected_team_in_league = false; }],
    ["fractional counts", (v: any) => { v.rows[0].team.possessions = 0.5; }],
    ["unsafe counts", (v: any) => { v.rows[0].team.possessions = Number.MAX_SAFE_INTEGER + 1; }],
    ["negative counts", (v: any) => { v.rows[0].league.field_goal_attempts = -1; }],
    ["NaN rate", (v: any) => { v.rows[0].team.fg_pct = NaN; }],
    ["out-of-range rate", (v: any) => { v.rows[0].team.ft_pct = 1.1; }],
    ["out-of-range eFG", (v: any) => { v.rows[0].team.efg_pct = 1.6; }],
    ["invented zero", (v: any) => { v.rows[0].team.field_goal_attempts = 0; v.rows[0].team.fg_pct = 0; }],
    ["missing rate with attempts", (v: any) => { v.rows[0].team.fg_pct = null; }],
    ["relative percent change", (v: any) => { v.rows[0].difference_pp.fg_pct = -100 / 6; }],
    ["NaN difference", (v: any) => { v.rows[0].difference_pp.fg_pct = NaN; }],
    ["difference without a benchmark", (v: any) => { v.rows[0].league = null; }],
    ["duplicate category", (v: any) => { v.rows.push(v.rows[0]); }],
    ["empty category", (v: any) => { v.rows[0].play_type = " "; }],
    ["no-data status with values", (v: any) => { v.status = "no_data"; }],
    ["success without rows", (v: any) => { v.rows = []; }],
  ])("rejects %s", (_name, mutate) => {
    const value = report(); (mutate as (v: any) => void)(value);
    expect(() => parseLeagueComparison(value, filters)).toThrow("incomplete");
  });
});
