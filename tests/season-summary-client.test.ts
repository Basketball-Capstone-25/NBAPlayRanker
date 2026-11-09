import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseSeasonSummary, summaryKey, type SeasonSummary, type SummaryFilters } from "../app/domain/season-summary";
import { emptySummary, fetchSeasonSummary, selectSummary, SummaryRequest, type SummaryState } from "../app/infrastructure/season-summary";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
const filters: SummaryFilters = { season: "2024-25", our: "TOR", opp: "BOS", k: 3, w_off: 0.7 };
const report = (): SeasonSummary => ({
  status: "ok", message: null,
  filters: { ...filters, period: "season", w_def: 1 - filters.w_off, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "regular-season-v1",
  summary: { observed_team_ppp: 1.7, modeled_topk_ppp: 0.9, modeled_uplift_ppp: -0.8,
    team_possessions: 400, selected_possessions: 100, selected_usage_pct: 25, k_returned: 1 },
  play_types: [{ rank: 1, play_type: "Cut", possessions: 100, usage_pct: 25 }],
  limitations: ["Season totals only.", "Historical comparison, not a future forecast."],
});
const noData = (): SeasonSummary => ({ ...report(), status: "no_data", message: "No data for this period.", summary: null, play_types: [] });
const deferred = <T>() => {
  let resolve!: (value: T) => void, reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

describe("season summary contract", () => {
  it("keeps negative uplift, fewer returned plays and whole-season usage", () => {
    const data = parseSeasonSummary(report(), filters);
    expect(data.status).toBe("ok");
    if (data.status !== "ok") throw new Error("Expected summary");
    expect(data.summary.modeled_uplift_ppp).toBe(-0.8);
    expect(data.summary.k_returned).toBe(1);
    expect(data.play_types[0].usage_pct).toBe(25);
  });
  it("accepts an empty period without inventing zero performance", () => {
    expect(parseSeasonSummary(noData(), filters)).toEqual(noData());
  });
  it.each(["season", "our", "opp", "k", "w_off"] as const)("rejects an old %s selection", field => {
    const expected = { ...filters, [field]: typeof filters[field] === "number" ? 1 : "different" };
    expect(() => parseSeasonSummary(report(), expected)).toThrow("incomplete");
  });
  it.each([
    ["recent-game scope", (v: any) => { v.filters.period = "last10"; }],
    ["playoff scope", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["defense scope", (v: any) => { v.filters.side = "defense"; }],
    ["wrong defense weight", (v: any) => { v.filters.w_def = 0.7; }],
    ["missing source", (v: any) => { v.dataset_version = ""; }],
    ["missing limitations", (v: any) => { v.limitations = []; }],
    ["wrong uplift", (v: any) => { v.summary.modeled_uplift_ppp = 0.8; }],
    ["non-finite PPP", (v: any) => { v.summary.observed_team_ppp = NaN; }],
    ["negative modeled PPP", (v: any) => { v.summary.modeled_topk_ppp = -1; }],
    ["zero denominator", (v: any) => { v.summary.team_possessions = 0; }],
    ["shortlist denominator", (v: any) => { v.play_types[0].usage_pct = 100; }],
    ["wrong selected usage", (v: any) => { v.summary.selected_usage_pct = 100; }],
    ["wrong selected possessions", (v: any) => { v.summary.selected_possessions = 50; }],
    ["more than full-season possessions", (v: any) => { v.summary.selected_possessions = 500; }],
    ["incorrect rank", (v: any) => { v.play_types[0].rank = 2; }],
    ["incorrect result count", (v: any) => { v.summary.k_returned = 2; }],
    ["duplicate play", (v: any) => {
      v.play_types.push({ ...v.play_types[0], rank: 2 });
      v.summary.k_returned = 2; v.summary.selected_possessions = 200; v.summary.selected_usage_pct = 50;
    }],
    ["empty status with numbers", (v: any) => { v.status = "no_data"; v.message = "No data"; }],
  ])("rejects %s", (_name, mutate) => {
    const value = report(); (mutate as (v: any) => void)(value);
    expect(() => parseSeasonSummary(value, filters)).toThrow("incomplete");
  });
});

describe("authenticated summary requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("sends the token only in the header and preserves filters and cancellation", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchSeasonSummary(filters, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    const query = new URL(url).searchParams;
    expect(url).toContain("/analytics/playtypes/summary?"); expect(url).not.toContain("test-token");
    expect(Object.fromEntries(query)).toEqual({ period: "season", season: "2024-25", our: "TOR", opp: "BOS", k: "3", w_off: "0.7" });
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it("does not request private data without a session or after failed session retrieval", async () => {
    getSession.mockResolvedValueOnce({ data: { session: null }, error: null })
      .mockRejectedValueOnce(new Error("private session detail"));
    await expect(fetchSeasonSummary(filters)).rejects.toThrow("Sign in again");
    await expect(fetchSeasonSummary(filters)).rejects.toThrow("Sign in again");
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([[401, "Sign in again"], [403, "coach or analyst"], [400, "Check the teams"],
    [422, "Check the teams"], [404, "unavailable"], [503, "unavailable"]])("handles HTTP %s without exposing server details", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("/private/server/detail", { status: Number(status) }));
    await expect(fetchSeasonSummary(filters)).rejects.toThrow(String(message));
    await expect(fetchSeasonSummary(filters)).rejects.not.toThrow("/private/");
  });
  it("handles connection failure, invalid JSON and mismatched successful data", async () => {
    const wrong = report(); wrong.filters.opp = "LAL";
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(wrong)));
    await expect(fetchSeasonSummary(filters)).rejects.toThrow("Could not reach");
    await expect(fetchSeasonSummary(filters)).rejects.toThrow("incomplete");
    await expect(fetchSeasonSummary(filters)).rejects.toThrow("incomplete");
  });
  it("stops before fetching if cancelled while the session was loading", async () => {
    const session = deferred<any>(), controller = new AbortController();
    getSession.mockReturnValue(session.promise);
    const run = fetchSeasonSummary(filters, controller.signal);
    controller.abort(); session.resolve({ data: { session: { access_token: "test-token" } } });
    await expect(run).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("summary request state", () => {
  let state: SummaryState;
  const publish = (next: SummaryState) => { state = next; };
  beforeEach(() => { state = emptySummary(); });
  afterEach(() => vi.useRealTimers());
  it("hides stale numbers during the render before the next request", async () => {
    await new SummaryRequest().run(filters, async () => report(), publish);
    expect(selectSummary(state, filters).data).toEqual(report());
    for (const [key, value] of Object.entries({ season: "2023-24", our: "LAL", opp: "DEN", k: 4, w_off: 0.5 })) {
      expect(selectSummary(state, { ...filters, [key]: value })).toEqual(emptySummary());
    }
  });
  it("clears previous results on refresh and failure, then permits retry", async () => {
    const request = new SummaryRequest();
    await request.run(filters, async () => report(), publish);
    const next = deferred<SeasonSummary>(), run = request.run(filters, () => next.promise, publish);
    expect(state.data).toBeNull(); expect(state.loading).toBe(true);
    next.reject(new Error("Summary unavailable")); await run;
    expect(state).toEqual({ key: summaryKey(filters), data: null, loading: false, error: "Summary unavailable" });
    await request.run(filters, async () => report(), publish);
    expect(state.data).toEqual(report()); expect(state.error).toBeNull();
  });
  it("ignores an old success or failure even if fetch ignores cancellation", async () => {
    for (const fail of [false, true]) {
      const request = new SummaryRequest(), old = deferred<SeasonSummary>();
      const run = request.run(filters, () => old.promise, publish);
      const next = { ...filters, k: 4 };
      await request.run(next, async () => report(), publish);
      await run;
      if (fail) old.reject(new Error("Old failure")); else old.resolve(report());
      await Promise.resolve();
      expect(state.key).toBe(summaryKey(next)); expect(state.error).toBeNull(); expect(state.loading).toBe(false);
    }
  });
  it("settles a hung request at the timeout and prevents a late result from replacing retry", async () => {
    vi.useFakeTimers();
    const request = new SummaryRequest(), slow = deferred<SeasonSummary>();
    const run = request.run(filters, () => slow.promise, publish);
    await vi.advanceTimersByTimeAsync(60000); await run;
    expect(state.error).toContain("took too long"); expect(state.data).toBeNull(); expect(state.loading).toBe(false);
    await request.run(filters, async () => noData(), publish);
    slow.resolve(report()); await Promise.resolve();
    expect(state.data?.status).toBe("no_data"); expect(state.error).toBeNull(); expect(vi.getTimerCount()).toBe(0);
  });
  it("cancels without publishing after unmount and clears its timer", async () => {
    vi.useFakeTimers();
    const request = new SummaryRequest(), slow = deferred<SeasonSummary>(), writes = vi.fn();
    const run = request.run(filters, () => slow.promise, writes);
    request.cancel(); await run;
    slow.resolve(report()); await Promise.resolve(); await vi.advanceTimersByTimeAsync(60000);
    expect(writes).toHaveBeenCalledTimes(1); expect(vi.getTimerCount()).toBe(0);
  });
  it("keeps an immutable filter snapshot and catches synchronous fetch errors", async () => {
    const current = { ...filters }, request = new SummaryRequest(), slow = deferred<SeasonSummary>();
    const fetchResult = vi.fn((_snapshot: SummaryFilters) => slow.promise);
    const run = request.run(current, fetchResult, publish);
    current.k = 9; slow.resolve(report()); await run;
    expect(fetchResult.mock.calls[0][0]).toEqual(filters); expect(state.key).toBe(summaryKey(filters));
    await request.run(filters, () => { throw new Error("Unavailable"); }, publish);
    expect(state.data).toBeNull(); expect(state.error).toBe("Unavailable"); expect(state.loading).toBe(false);
  });
});
