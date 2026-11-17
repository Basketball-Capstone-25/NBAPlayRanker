import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OUTCOME_METRICS, type PossessionOutcomes } from "../app/domain/possession-outcomes";
import { fetchPossessionOutcomes } from "../app/infrastructure/possession-outcomes";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
const filters = { season: "2024-25", team: "TOR" };
const report = (): PossessionOutcomes => ({
  status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "synergy_regular_season_v2",
  rows: [{ play_type: "Cut", possessions: 100, rates: Object.fromEntries(OUTCOME_METRICS.map(({ key }) =>
    [key, { value: 0.3, status: "ok", usable_possessions: 100, missing_possessions: 0, invalid_possessions: 0 }])) as PossessionOutcomes["rows"][number]["rates"] }],
});

describe("possession outcome requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("sends only team and season with the token in a header", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchPossessionOutcomes({ ...filters, side: "defense" } as typeof filters, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe("/analytics/playtypes/possession-outcomes");
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual(filters);
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it("does not fetch with a missing or failed session", async () => {
    getSession.mockResolvedValueOnce({ data: { session: null } }).mockRejectedValueOnce(new Error("private detail"))
      .mockResolvedValueOnce({ data: { session: { access_token: "test-token" } }, error: new Error("private detail") });
    for (let i = 0; i < 3; i++) await expect(fetchPossessionOutcomes(filters)).rejects.toThrow("Sign in again");
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([[401, "Sign in again"], [403, "coach account"], [400, "available season"],
    [422, "available season"], [404, "unavailable"], [503, "unavailable"]])("explains HTTP %s without server details", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("/private/server/detail", { status: Number(status) }));
    await expect(fetchPossessionOutcomes(filters)).rejects.toThrow(String(message));
  });
  it("rejects network errors, invalid JSON and a different selection", async () => {
    const wrong = report(); wrong.filters.team = "BOS";
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(wrong)));
    await expect(fetchPossessionOutcomes(filters)).rejects.toThrow("Could not reach");
    await expect(fetchPossessionOutcomes(filters)).rejects.toThrow("incomplete");
    await expect(fetchPossessionOutcomes(filters)).rejects.toThrow("incomplete");
  });
  it("preserves unavailable rates without inventing zeros", async () => {
    const value = report(); value.rows[0].rates.scoring = {
      value: null, status: "missing", usable_possessions: 80, missing_possessions: 20, invalid_possessions: 0,
    };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(value)));
    await expect(fetchPossessionOutcomes(filters)).resolves.toEqual(value);
  });
  it("snapshots the selection before asynchronous session lookup", async () => {
    const mutable = { ...filters };
    getSession.mockImplementationOnce(async () => {
      mutable.team = "BOS"; mutable.season = "2023-24";
      return { data: { session: { access_token: "test-token" } } };
    });
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    await expect(fetchPossessionOutcomes(mutable)).resolves.toEqual(report());
    expect(Object.fromEntries(new URL(fetchMock.mock.calls[0][0]).searchParams)).toEqual(filters);
  });
  it("stops before fetching if canceled before or during session lookup", async () => {
    const first = new AbortController(); first.abort();
    await expect(fetchPossessionOutcomes(filters, first.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(getSession).not.toHaveBeenCalled();
    const next = new AbortController();
    getSession.mockImplementationOnce(async () => { next.abort(); return { data: { session: null } }; });
    await expect(fetchPossessionOutcomes(filters, next.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("preserves cancellation during the request", async () => {
    const controller = new AbortController();
    fetchMock.mockImplementationOnce(async () => { controller.abort(); throw controller.signal.reason; });
    await expect(fetchPossessionOutcomes(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
  it("drops a reply canceled while its JSON is being read", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => { controller.abort(); return report(); } });
    await expect(fetchPossessionOutcomes(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
});
