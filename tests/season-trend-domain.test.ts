import { describe, expect, it } from "vitest";
import { parseSeasonTrend, trendKey, type SeasonTrend } from "../app/domain/season-trend";

const filters = { team: "TOR", play_type: "Cut" };
const report = (): SeasonTrend => ({
  status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "regular-season",
  rows: [
    { season: "2022-23", possessions: 100, points: 120, team_possessions: 400, usage_pct: 25, ppp: 1.2 },
    { season: "2023-24", possessions: null, points: null, team_possessions: 500, usage_pct: null, ppp: null },
    { season: "2024-25", possessions: 0, points: 0, team_possessions: 600, usage_pct: 0, ppp: null },
  ],
});

describe("season trend contract", () => {
  it("keeps a missing category and recorded zero distinct", () => {
    expect(parseSeasonTrend(report(), filters)).toEqual(report());
    expect(trendKey(filters)).not.toBe(trendKey({ ...filters, play_type: "Spotup" }));
  });
  it("accepts missing team seasons and a no-data result with season gaps", () => {
    const data = report(); data.status = "no_data";
    data.rows = [data.rows[1]]; data.rows[0].team_possessions = null;
    expect(parseSeasonTrend(data, filters)).toEqual(data);
    data.rows = [report().rows[2]];
    expect(parseSeasonTrend(data, filters)).toEqual(data);
    data.rows[0].team_possessions = 0; data.rows[0].usage_pct = null;
    expect(parseSeasonTrend(data, filters)).toEqual(data);
  });
  it.each([
    ["other team", (v: any) => { v.filters.team = "BOS"; }],
    ["other play", (v: any) => { v.filters.play_type = "Spotup"; }],
    ["defense", (v: any) => { v.filters.side = "defense"; }],
    ["playoffs", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["missing source", (v: any) => { v.dataset_version = ""; }],
    ["invalid year", (v: any) => { v.rows[0].season = "2022-99"; }],
    ["omitted season", (v: any) => { v.rows.splice(1, 1); }],
    ["duplicate season", (v: any) => { v.rows[1].season = "2022-23"; }],
    ["reverse order", (v: any) => { v.rows.reverse(); }],
    ["fractional count", (v: any) => { v.rows[0].possessions = 100.5; }],
    ["unsafe count", (v: any) => { v.rows[0].points = Number.MAX_SAFE_INTEGER + 1; }],
    ["negative total", (v: any) => { v.rows[0].team_possessions = -1; }],
    ["missing denominator", (v: any) => { v.rows[0].team_possessions = null; }],
    ["wrong usage", (v: any) => { v.rows[0].usage_pct = 20; }],
    ["wrong PPP", (v: any) => { v.rows[0].ppp = 1.5; }],
    ["non-finite rate", (v: any) => { v.rows[0].ppp = NaN; }],
    ["invented gap zero", (v: any) => { v.rows[1].usage_pct = 0; }],
    ["points without possessions", (v: any) => { v.rows[2].points = 2; }],
    ["zero PPP without possessions", (v: any) => { v.rows[2].ppp = 0; }],
    ["wrong status", (v: any) => { v.status = "no_data"; }],
    ["no calendar rows", (v: any) => { v.rows = []; }],
  ])("rejects %s", (_name, mutate) => {
    const data = report(); (mutate as (v: any) => void)(data);
    expect(() => parseSeasonTrend(data, filters)).toThrow("incomplete");
  });
});
