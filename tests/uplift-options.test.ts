import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
import { fetchMetaOptions } from "../app/infrastructure/api-client";

describe("Top-K uplift options", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "analyst-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("marks a failed options request so the panel can offer a retry", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const meta = await fetchMetaOptions();
    expect(meta._fallback).toBe(true);
    expect(meta.seasons).toEqual([]);
  });

  it("returns the real lists when the request works", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ seasons: ["2024-25"], teams: ["BOS", "TOR"] }), { status: 200 }));
    const meta = await fetchMetaOptions();
    expect(meta._fallback).toBeUndefined();
    expect(meta.seasons).toEqual(["2024-25"]);
  });
});
