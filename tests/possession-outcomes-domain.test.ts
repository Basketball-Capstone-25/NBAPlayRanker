import { describe, expect, it } from "vitest";
import { OUTCOME_METRICS, outcomeKey, parsePossessionOutcomes, type OutcomeRate, type PossessionOutcomes } from "../app/domain/possession-outcomes";

const filters = { season: "2019-20", team: "PHX" };
const rate = (value: number): OutcomeRate => ({ value, status: "ok", usable_possessions: 486, missing_possessions: 0, invalid_possessions: 0 });
// Possession-weighted PHX Misc rates from the 2019-20 source rows.
const report = (): PossessionOutcomes => ({
  status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "synergy_regular_season_v2",
  rows: [{ play_type: "Misc", possessions: 486, rates: {
    scoring: rate(0.34764197530864194), turnover: rate(0.5287901234567901), free_throw: rate(0.27368930041152256),
    shooting_foul: rate(0.030788065843621395), and_one: rate(0.006189300411522634),
  } }],
});

describe("possession outcome contract", () => {
  it("preserves overlapping source rates above 100% without normalizing", () => {
    const value = report();
    const total = Object.values(value.rows[0].rates).reduce((sum, metric) => sum + metric.value!, 0);
    expect(total).toBeGreaterThan(1);
    expect(parsePossessionOutcomes(value, filters)).toEqual(value);
    expect(OUTCOME_METRICS).toHaveLength(5);
    expect(outcomeKey(filters)).not.toBe(outcomeKey({ ...filters, team: "TOR" }));
    expect(outcomeKey(filters)).not.toBe(outcomeKey({ ...filters, season: "2024-25" }));
  });
  it("distinguishes a measured zero from missing and invalid source rates", () => {
    const value = report(), rates = value.rows[0].rates;
    rates.scoring.value = 0;
    rates.turnover = { value: null, status: "missing", usable_possessions: 400, missing_possessions: 86, invalid_possessions: 0 };
    rates.free_throw = { value: null, status: "invalid", usable_possessions: 400, missing_possessions: 60, invalid_possessions: 26 };
    expect(parsePossessionOutcomes(value, filters)).toEqual(value);
  });
  it("keeps zero-possession rows while treating an absent season as no data", () => {
    const value = report(); value.rows[0].possessions = 0;
    for (const key of OUTCOME_METRICS.map(metric => metric.key)) value.rows[0].rates[key] = {
      value: null, status: "no_possessions", usable_possessions: 0, missing_possessions: 0, invalid_possessions: 0,
    };
    expect(parsePossessionOutcomes(value, filters).status).toBe("ok");
    value.status = "no_data"; value.rows = [];
    expect(parsePossessionOutcomes(value, filters).rows).toEqual([]);
  });
  it.each([
    ["wrong season", (v: any) => { v.filters.season = "2024-25"; }],
    ["wrong team", (v: any) => { v.filters.team = "TOR"; }],
    ["defensive data", (v: any) => { v.filters.side = "defense"; }],
    ["playoffs", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["missing source", (v: any) => { v.dataset_version = ""; }],
    ["missing aggregation", (v: any) => { delete v.aggregation_version; }],
    ["empty success", (v: any) => { v.rows = []; }],
    ["no-data with rows", (v: any) => { v.status = "no_data"; }],
    ["duplicate play type", (v: any) => { v.rows.push(v.rows[0]); }],
    ["empty play type", (v: any) => { v.rows[0].play_type = " "; }],
    ["long play type", (v: any) => { v.rows[0].play_type = "x".repeat(41); }],
    ["too many play types", (v: any) => { v.rows = Array.from({ length: 31 }, (_, i) => ({ ...v.rows[0], play_type: `Play${i}` })); }],
    ["negative count", (v: any) => { v.rows[0].possessions = -1; }],
    ["fractional count", (v: any) => { v.rows[0].possessions = 486.5; }],
    ["unsafe count", (v: any) => { v.rows[0].possessions = Number.MAX_SAFE_INTEGER + 1; }],
    ["boolean count", (v: any) => { v.rows[0].possessions = true; }],
    ["absent metric", (v: any) => { delete v.rows[0].rates.scoring; }],
    ["missing coverage count", (v: any) => { delete v.rows[0].rates.scoring.invalid_possessions; }],
    ["negative coverage", (v: any) => { v.rows[0].rates.scoring.missing_possessions = -1; }],
    ["fractional coverage", (v: any) => { v.rows[0].rates.scoring.usable_possessions = 485.5; }],
    ["incomplete coverage", (v: any) => { v.rows[0].rates.scoring.usable_possessions = 485; }],
    ["excess coverage", (v: any) => { v.rows[0].rates.scoring.invalid_possessions = 1; }],
    ["unsafe coverage sum", (v: any) => { v.rows[0].rates.scoring.usable_possessions = Number.MAX_SAFE_INTEGER; v.rows[0].rates.scoring.invalid_possessions = 1; }],
    ["unknown status", (v: any) => { v.rows[0].rates.scoring.status = "available"; }],
    ["false missing status", (v: any) => { v.rows[0].rates.scoring.status = "missing"; v.rows[0].rates.scoring.value = null; }],
    ["missing usable value", (v: any) => { v.rows[0].rates.scoring.value = null; }],
    ["negative rate", (v: any) => { v.rows[0].rates.scoring.value = -0.1; }],
    ["rate above one", (v: any) => { v.rows[0].rates.scoring.value = 1.01; }],
    ["nonfinite rate", (v: any) => { v.rows[0].rates.scoring.value = Infinity; }],
    ["NaN rate", (v: any) => { v.rows[0].rates.scoring.value = NaN; }],
    ["string rate", (v: any) => { v.rows[0].rates.scoring.value = "0.3"; }],
  ])("rejects %s", (_name, change) => {
    const value = report(); (change as (v: any) => void)(value);
    expect(() => parsePossessionOutcomes(value, filters)).toThrow("incomplete");
  });
  it("requires invalid to take precedence over missing and hides partial averages", () => {
    const value = report(), metric = value.rows[0].rates.scoring;
    Object.assign(metric, { value: null, status: "missing", usable_possessions: 400, missing_possessions: 60, invalid_possessions: 26 });
    expect(() => parsePossessionOutcomes(value, filters)).toThrow("incomplete");
    metric.status = "invalid"; metric.value = 0.3;
    expect(() => parsePossessionOutcomes(value, filters)).toThrow("incomplete");
    metric.value = null;
    expect(parsePossessionOutcomes(value, filters)).toEqual(value);
  });
});
