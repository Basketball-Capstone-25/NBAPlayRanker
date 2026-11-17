import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TeamComparison, TeamComparisonFilters } from "../app/domain/team-comparison";
import { fetchTeamComparison } from "../app/infrastructure/team-comparison";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

const filters: TeamComparisonFilters = { season: "2024-25", team: "TOR", opponent: "BOS" };
const report = (): TeamComparison => ({
  status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "synergy_regular_season_v2",
  team_possessions: 100, opponent_possessions: 200,
  rows: [{ play_type: "Cut",
    team: { possessions: 100, points: 120, usage_pct: 100, ppp: 1.2 },
    opponent: { possessions: 200, points: 200, usage_pct: 100, ppp: 1 },
    difference: { usage_pp: 0, ppp: 0.2 },
  }],
});

describe("team comparison requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("sends only both teams and the season, with the token in a header", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    const selection = { ...filters, side: "defense", top_k: 3 };
    await expect(fetchTeamComparison(selection, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe("/analytics/playtypes/team-comparison");
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual(filters);
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it("does not fetch without a session or when session lookup fails", async () => {
    getSession.mockResolvedValueOnce({ data: { session: null } }).mockRejectedValueOnce(new Error("private detail"))
      .mockResolvedValueOnce({ data: { session: { access_token: "test-token" } }, error: new Error("private detail") });
    for (let request = 0; request < 3; request++) await expect(fetchTeamComparison(filters)).rejects.toThrow("Sign in again");
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([[401, "Sign in again"], [403, "coach account"], [400, "two different teams"],
    [422, "two different teams"], [404, "unavailable"], [503, "unavailable"]])("explains HTTP %s without server details", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("/private/server/detail", { status: Number(status) }));
    await expect(fetchTeamComparison(filters)).rejects.toThrow(String(message));
  });
  it("rejects connection errors, invalid JSON and replies for a different opponent", async () => {
    const wrong = report(); wrong.filters.opponent = "LAL";
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(wrong)));
    await expect(fetchTeamComparison(filters)).rejects.toThrow("Could not reach");
    await expect(fetchTeamComparison(filters)).rejects.toThrow("incomplete");
    await expect(fetchTeamComparison(filters)).rejects.toThrow("incomplete");
  });
  it("preserves a missing team profile without filling it with zeros", async () => {
    const missing = report(); missing.status = "no_data"; missing.opponent_possessions = null;
    missing.rows[0].opponent = null; missing.rows[0].difference = { usage_pp: null, ppp: null };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(missing)));
    await expect(fetchTeamComparison(filters)).resolves.toEqual(missing);
  });
  it("uses the initial selection if it changes during session lookup", async () => {
    const mutable = { ...filters };
    getSession.mockImplementationOnce(async () => {
      mutable.team = "BOS"; mutable.opponent = "TOR"; mutable.season = "2023-24";
      return { data: { session: { access_token: "test-token" } } };
    });
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    await expect(fetchTeamComparison(mutable)).resolves.toEqual(report());
    expect(Object.fromEntries(new URL(fetchMock.mock.calls[0][0]).searchParams)).toEqual(filters);
  });
  it("stops if canceled before or during session lookup", async () => {
    const controller = new AbortController(); controller.abort();
    await expect(fetchTeamComparison(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(getSession).not.toHaveBeenCalled();
    const next = new AbortController();
    getSession.mockImplementationOnce(async () => { next.abort(); return { data: { session: null } }; });
    await expect(fetchTeamComparison(filters, next.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("preserves cancellation during the request", async () => {
    const controller = new AbortController();
    fetchMock.mockImplementationOnce(async () => { controller.abort(); throw controller.signal.reason; });
    await expect(fetchTeamComparison(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
  it("does not return numbers if canceled while parsing the response", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => { controller.abort(); return report(); } });
    await expect(fetchTeamComparison(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
});
