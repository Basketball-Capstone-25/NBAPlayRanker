import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
import { fetchTopKUplift, topKUpliftUrl, UpliftUnavailableError } from "../app/infrastructure/topk-uplift";

describe("Top-K uplift client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "analyst-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("builds the uplift URL with the matchup filters", () => {
    const url = new URL(topKUpliftUrl({ season: "2024-25", our: "TOR", opp: "BOS", k: 5 }), "https://api.test");
    expect(url.pathname).toMatch(/\/metrics\/topk-uplift$/);
    expect(Object.fromEntries(url.searchParams)).toEqual({ season: "2024-25", our: "TOR", opp: "BOS", k: "5", w_off: "0.7" });
  });

  it("sends the signed-in token and returns the report", async () => {
    const report = { uplift_ppp: 0.05, rankings: [] };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report), { status: 200 }));
    await expect(fetchTopKUplift({ season: "2024-25", our: "TOR", opp: "BOS", k: 5 })).resolves.toEqual(report);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer analyst-token" });
  });

  it("reports a missing matchup as unavailable, not as a failure", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "No data for this matchup" }), { status: 404 }));
    await expect(fetchTopKUplift({ season: "2019-20", our: "TOR", opp: "BOS", k: 5 })).rejects.toBeInstanceOf(UpliftUnavailableError);
  });

  it("surfaces other API errors with their message", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "Forbidden" }), { status: 403 }));
    await expect(fetchTopKUplift({ season: "2024-25", our: "TOR", opp: "BOS", k: 5 })).rejects.toThrow("Forbidden");
  });
});
