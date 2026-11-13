import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchSeasonTrend } from "../app/infrastructure/season-trend";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
const filters = { team: "TOR", play_type: "Cut" };
const report = () => ({ status: "ok", filters: { ...filters, side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture", aggregation_version: "regular-season", rows: [
    { season: "2024-25", possessions: 100, points: 120, team_possessions: 400, usage_pct: 25, ppp: 1.2 }],
});

describe("season trend requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("sends the selected team and play type with a bearer header", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchSeasonTrend(filters, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe("/analytics/playtypes/season-trend");
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual(filters);
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it("keeps a snapshot if the selection changes during session lookup", async () => {
    const mutable = { ...filters };
    getSession.mockImplementationOnce(async () => {
      mutable.play_type = "Spotup";
      return { data: { session: { access_token: "test-token" } } };
    });
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    await expect(fetchSeasonTrend(mutable)).resolves.toEqual(report());
    expect(new URL(fetchMock.mock.calls[0][0]).searchParams.get("play_type")).toBe("Cut");
  });
  it("does not request data after failed session lookup or cancellation", async () => {
    getSession.mockRejectedValueOnce(new Error("private detail"));
    await expect(fetchSeasonTrend(filters)).rejects.toThrow("Sign in again");
    const controller = new AbortController(); controller.abort();
    await expect(fetchSeasonTrend(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([[401, "Sign in again"], [403, "coach account"], [400, "Check the team"],
    [422, "Check the team"], [503, "unavailable"]])("explains HTTP %s", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("private server detail", { status: Number(status) }));
    await expect(fetchSeasonTrend(filters)).rejects.toThrow(String(message));
  });
  it("rejects network failures, malformed JSON and changed scope", async () => {
    const wrong = report(); wrong.filters.team = "BOS";
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(wrong)));
    await expect(fetchSeasonTrend(filters)).rejects.toThrow("Could not reach");
    await expect(fetchSeasonTrend(filters)).rejects.toThrow("incomplete");
    await expect(fetchSeasonTrend(filters)).rejects.toThrow("incomplete");
  });
  it("rejects a reply canceled during JSON parsing", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => { controller.abort(); return report(); } });
    await expect(fetchSeasonTrend(filters, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
});
