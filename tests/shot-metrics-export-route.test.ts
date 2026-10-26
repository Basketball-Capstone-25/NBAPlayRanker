import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { createClient, getUser, getSession } = vi.hoisted(() => ({ createClient: vi.fn(), getUser: vi.fn(), getSession: vi.fn() }));
vi.mock("../lib/supabase/server", () => ({ createClient }));
import { GET } from "../app/api/exports/shot-metrics/route";

const version = "a".repeat(64);
const metric = { model: "gbr", RMSE_mean: 1.16, RMSE_std: .01, MAE_mean: 1.08, MAE_std: .02, R2_mean: -.04, R2_std: .01 };
const saved = { source_sha256: version, n_splits: 5, cached: true, metrics: [metric] };
function request(query = `version=${version}`, headers: HeadersInit = {}) {
  return new Request(`https://frontend.example.test/api/exports/shot-metrics?${query}`, { headers });
}

describe("saved shot metrics download", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "https://api.example.test");
    vi.stubGlobal("fetch", fetchMock);
    createClient.mockResolvedValue({ auth: { getUser, getSession } });
    getUser.mockResolvedValue({ data: { user: { id: "verified-analyst" } }, error: null });
    getSession.mockResolvedValue({ data: { session: { access_token: "private-fixture-token", user: { id: "verified-analyst" } } }, error: null });
    fetchMock.mockImplementation(async () => Response.json(saved));
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

  it("uses the verified cookie session and fixed five-fold route, then returns a versioned CSV", async () => {
    const response = await GET(request(undefined, { Authorization: "Bearer ignored-client-token" }));
    expect(getUser).toHaveBeenCalledOnce();
    expect(getSession).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toBe("https://api.example.test/metrics/shot-models?n_splits=5");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ headers: { Authorization: "Bearer private-fixture-token" }, cache: "no-store", redirect: "manual" });
    expect(response.status).toBe(200);
    expect(response.headers.get("content-disposition")).toBe('attachment; filename="shot_model_metrics_splits_5.csv"');
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(response.headers.get("x-dataset-version")).toBe(`sha256:${version}`);
    expect(await response.text()).toBe(`model,RMSE_mean,RMSE_std,MAE_mean,MAE_std,R2_mean,R2_std,n_splits,DATASET_VERSION\r\n"gbr",1.16,0.01,1.08,0.02,-0.04,0.01,5,"sha256:${version}"\r\n`);
  });

  it("rejects a missing or mismatched session before any upstream request", async () => {
    getUser.mockResolvedValueOnce({ data: { user: null }, error: null });
    expect((await GET(request())).status).toBe(401);
    getSession.mockResolvedValueOnce({ data: { session: { access_token: "wrong", user: { id: "different" } } }, error: null });
    expect((await GET(request())).status).toBe(401);
    getSession.mockResolvedValueOnce({ data: { session: null }, error: null });
    expect((await GET(request())).status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([{ Origin: "https://other.example.test" }, { "Sec-Fetch-Site": "cross-site" }])("rejects foreign requests", async headers => {
    expect((await GET(request(undefined, headers))).status).toBe(403);
    expect(createClient).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(["", "version=not-a-hash", `version=${version}&version=${version}`, `version=${version}&path=https://other.example.test`])("rejects arbitrary or invalid query parameters", async query => {
    expect((await GET(request(query))).status).toBe(400);
    expect(createClient).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([[401, 401], [403, 403], [503, 503], [500, 502], [302, 502]])("keeps upstream %i out of attachments", async (upstreamStatus, expected) => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: upstreamStatus }));
    const response = await GET(request());
    expect(response.status).toBe(expected);
    expect(response.headers.has("content-disposition")).toBe(false);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.text()).not.toContain("private-fixture-token");
  });

  it.each([
    { ...saved, n_splits: 3 }, { ...saved, cached: false }, { ...saved, source_sha256: "invalid" },
    { ...saved, metrics: [] }, { ...saved, metrics: Array.from({ length: 11 }, (_, i) => ({ ...metric, model: `model-${i}` })) },
    { ...saved, metrics: [metric, metric] }, { ...saved, metrics: [{ ...metric, MAE_mean: null }] },
    { ...saved, metrics: [{ ...metric, RMSE_mean: "1.1" }] }, { ...saved, metrics: [{ ...metric, R2_std: Infinity }] },
  ])("rejects incomplete, unbounded or nonfinite saved metrics", async body => {
    fetchMock.mockResolvedValueOnce(Response.json(body));
    const response = await GET(request());
    expect(response.status).toBe(502);
    expect(response.headers.has("content-disposition")).toBe(false);
  });

  it("requires reload when the saved source changed after display", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, source_sha256: "b".repeat(64) }));
    const response = await GET(request());
    expect(response.status).toBe(409);
    expect(response.headers.has("content-disposition")).toBe(false);
  });

  it("escapes CSV text while preserving negative numerical values", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, metrics: [{ ...metric, model: '=SUM(1,2)"' }] }));
    const text = await (await GET(request())).text();
    expect(text).toContain('"\'=SUM(1,2)"""');
    expect(text).toContain(",-0.04,");
  });

  it("rejects HTML, malformed JSON and oversized bodies", async () => {
    for (const response of [new Response("<html>Error</html>", { headers: { "content-type": "text/html" } }),
      new Response("{", { headers: { "content-type": "application/json" } }),
      new Response(" ".repeat(256 * 1024 + 1), { headers: { "content-type": "application/json" } })]) {
      fetchMock.mockResolvedValueOnce(response);
      const result = await GET(request());
      expect(result.status).toBe(502);
      expect(result.headers.has("content-disposition")).toBe(false);
    }
  });

  it("returns a plain error after a network failure", async () => {
    fetchMock.mockRejectedValueOnce(new Error("private-fixture-token"));
    const result = await GET(request());
    expect(result.status).toBe(502);
    expect(result.headers.has("content-disposition")).toBe(false);
    expect(await result.text()).not.toContain("private-fixture-token");
  });
});
