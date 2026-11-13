import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { leagueKey, parseLeagueComparison, type LeagueComparison, type LeagueFilters } from "../app/domain/league-comparison";
import { fetchLeagueComparison } from "../app/infrastructure/league-comparison";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

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

describe("league comparison requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("sends only the selected scope and keeps the token in a header", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchLeagueComparison(filters, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe("/analytics/playtypes/league-comparison");
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual(filters);
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it("does not fetch without a session or when session lookup fails", async () => {
    getSession.mockResolvedValueOnce({ data: { session: null } }).mockRejectedValueOnce(new Error("private detail"));
    await expect(fetchLeagueComparison(filters)).rejects.toThrow("Sign in again");
    await expect(fetchLeagueComparison(filters)).rejects.toThrow("Sign in again");
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([[401, "Sign in again"], [403, "coach account"], [400, "Check the team"],
    [422, "Check the team"], [404, "unavailable"], [503, "unavailable"]])("explains HTTP %s without server details", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("/private/server/detail", { status: Number(status) }));
    await expect(fetchLeagueComparison(filters)).rejects.toThrow(String(message));
  });
  it("rejects connection failures, invalid JSON and mismatched successful data", async () => {
    const wrong = report(); wrong.filters.team = "LAL";
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(wrong)));
    await expect(fetchLeagueComparison(filters)).rejects.toThrow("Could not reach");
    await expect(fetchLeagueComparison(filters)).rejects.toThrow("incomplete");
    await expect(fetchLeagueComparison(filters)).rejects.toThrow("incomplete");
  });
  it("uses the initial filters if the caller changes them during session lookup", async () => {
    const mutable = { ...filters };
    getSession.mockImplementationOnce(async () => {
      mutable.team = "BOS";
      return { data: { session: { access_token: "test-token" } } };
    });
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    await expect(fetchLeagueComparison(mutable)).resolves.toEqual(report());
    expect(new URL(fetchMock.mock.calls[0][0]).searchParams.get("team")).toBe("TOR");
  });
  it("stops if canceled before or during session lookup", async () => {
    const controller = new AbortController(); controller.abort();
    await expect(fetchLeagueComparison(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(getSession).not.toHaveBeenCalled();
    const next = new AbortController();
    getSession.mockImplementationOnce(async () => { next.abort(); return { data: { session: null } }; });
    await expect(fetchLeagueComparison(filters, next.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("does not return numbers if canceled while parsing the response", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => { controller.abort(); return report(); } });
    await expect(fetchLeagueComparison(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
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
