import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchPbpMetaOptions } from "../app/infrastructure/api-client";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: {
  getSession: async () => ({ data: { session: null } }),
} }) }));

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("strict shot metadata refresh", () => {
  it("returns current shot options without requesting another dataset", async () => {
    const options = { seasons: ["2024-25"], teams: ["TOR"], shotTypes: ["2PT"], zones: ["Restricted Area"] };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(options)));
    await expect(fetchPbpMetaOptions({ strict: true })).resolves.toMatchObject(options);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain("/pbp/meta/options");
  });
  it("rejects an unavailable shot source instead of inventing fallback options", async () => {
    fetchMock.mockResolvedValue(new Response("unavailable", { status: 503 }));
    await expect(fetchPbpMetaOptions({ strict: true })).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it("preserves the existing fallback for callers that did not request strict refresh", async () => {
    fetchMock.mockResolvedValueOnce(new Response("unavailable", { status: 503 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ seasons: ["2024-25"], teams: ["TOR"] })));
    await expect(fetchPbpMetaOptions()).resolves.toEqual({ seasons: ["2024-25"], teams: ["TOR"], shotTypes: [], zones: [] });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
