import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseShotHeatmapExport, shotHeatmapExportUrl } from "../app/domain/shot-heatmap-export";

const { createClient, getUser, getSession, maybeSingle, from, select, eq } = vi.hoisted(() => ({
  createClient: vi.fn(), getUser: vi.fn(), getSession: vi.fn(), maybeSingle: vi.fn(), from: vi.fn(), select: vi.fn(), eq: vi.fn(),
}));
vi.mock("../lib/supabase/server", () => ({ createClient }));
import { GET } from "../app/api/exports/shot-heatmap/route";

const version = `sha256:${"a".repeat(64)}`;
const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aC3sAAAAASUVORK5CYII=";
const inputs = { season: "2025-26", team: "OKC", opp: "SAS", shotType: "Stepback Jump Shot", zone: "above_break_3", maxShots: 1000 };
const saved = { season: inputs.season, team: inputs.team, opp: inputs.opp, shot_type: inputs.shotType, zone: inputs.zone,
  max_shots: inputs.maxShots, dataset_version: version, n_shots_total: 1083, n_shots_rendered: 1000,
  n_shots: 1000, n_shots_excluded_location: 1, n_shots_outside_halfcourt: 2, n_shots_omitted_by_cap: 80, image_base64: png };
function request(query?: string, headers: HeadersInit = {}) {
  return new Request(query === undefined ? `https://frontend.example.test${shotHeatmapExportUrl(inputs, version)}` :
    `https://frontend.example.test/api/exports/shot-heatmap?${query}`, { headers });
}
function params() { return new URL(request().url).searchParams; }

describe("shot heatmap attachment", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "https://api.example.test");
    vi.stubGlobal("fetch", fetchMock);
    createClient.mockResolvedValue({ auth: { getUser, getSession }, from });
    getUser.mockResolvedValue({ data: { user: { id: "verified-user", user_metadata: { role: "analyst" } } }, error: null });
    getSession.mockResolvedValue({ data: { session: { access_token: "private-fixture-token", user: { id: "verified-user" } } }, error: null });
    from.mockReturnValue({ select }); select.mockReturnValue({ eq }); eq.mockReturnValue({ maybeSingle });
    maybeSingle.mockResolvedValue({ data: { role: "analyst" }, error: null });
    fetchMock.mockImplementation(async () => Response.json(saved));
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

  it.each(["analyst", "coach"])("downloads identical PNG bytes for the current %s role with the exact saved inputs", async role => {
    maybeSingle.mockResolvedValueOnce({ data: { role }, error: null });
    const response = await GET(request(undefined, { Authorization: "Bearer ignored-client-token" }));
    expect(response.status).toBe(200);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("https://api.example.test/pbp/viz/shot-heatmap?season=2025-26&our=OKC&opp=SAS&max_shots=1000&shot_type=Stepback+Jump+Shot&zone=above_break_3");
    expect(init).toMatchObject({ headers: { Authorization: "Bearer private-fixture-token" }, cache: "no-store", redirect: "manual" });
    expect(getUser).toHaveBeenCalledOnce(); expect(getSession).toHaveBeenCalledOnce();
    expect(from).toHaveBeenCalledWith("profiles"); expect(select).toHaveBeenCalledWith("role"); expect(eq).toHaveBeenCalledWith("id", "verified-user");
    expect(response.headers.get("content-type")).toBe("image/png");
    expect(response.headers.get("content-disposition")).toBe('attachment; filename="shot_heatmap_2025-26_OKC_vs_SAS_Stepback_Jump_Shot_above_break_3_max_1000.png"');
    expect(response.headers.get("content-length")).toBe(String(Buffer.from(png, "base64").length));
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(response.headers.get("x-dataset-version")).toBe(version);
    expect(response.headers.get("x-shots-matched")).toBe("1083"); expect(response.headers.get("x-shots-shown")).toBe("1000");
    expect(Buffer.from(await response.arrayBuffer())).toEqual(Buffer.from(png, "base64"));
  });

  it("accepts the unfiltered matchup and keeps absent filters out of the backend request", async () => {
    const url = shotHeatmapExportUrl({ ...inputs, shotType: "", zone: "" }, version)!;
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, shot_type: null, zone: null }));
    const response = await GET(new Request(`https://frontend.example.test${url}`));
    expect(response.status).toBe(200);
    expect(String(fetchMock.mock.calls[0][0])).not.toMatch(/shot_type|zone/);
    expect(response.headers.get("content-disposition")).toContain("alltypes_allzones_max_1000.png");
  });

  it("downloads a valid PNG when all matching attempts were excluded from the half court", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, caption: "No usable locations in the displayed half court",
      n_shots_total: 3, n_shots: 0, n_shots_rendered: 0, n_shots_excluded_location: 1,
      n_shots_outside_halfcourt: 2, n_shots_omitted_by_cap: 0 }));
    const response = await GET(request());
    expect(response.status).toBe(200);
    expect(response.headers.get("x-shots-matched")).toBe("3");
    expect(response.headers.get("x-shots-shown")).toBe("0");
    expect(response.headers.get("content-type")).toBe("image/png");
    expect(Buffer.from(await response.arrayBuffer())).toEqual(Buffer.from(png, "base64"));
  });

  it("rejects absent, expired and mismatched sessions before calling the backend", async () => {
    getUser.mockResolvedValueOnce({ data: { user: null }, error: null });
    expect((await GET(request())).status).toBe(401);
    getSession.mockResolvedValueOnce({ data: { session: null }, error: null });
    expect((await GET(request())).status).toBe(401);
    getSession.mockResolvedValueOnce({ data: { session: { access_token: "wrong", user: { id: "another-user" } } }, error: null });
    expect((await GET(request())).status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([null, { role: "pending" }, { role: "admin" }])("requires a permitted current database role, ignoring signup metadata", async profile => {
    maybeSingle.mockResolvedValueOnce({ data: profile, error: null });
    const response = await GET(request());
    expect(response.status).toBe(403); expect(response.headers.has("content-disposition")).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fails closed when the current role query fails", async () => {
    maybeSingle.mockResolvedValueOnce({ data: { role: "analyst" }, error: { message: "unavailable" } });
    expect((await GET(request())).status).toBe(403); expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([{ Origin: "https://other.example.test" }, { "Sec-Fetch-Site": "cross-site" }])("rejects foreign requests", async headers => {
    expect((await GET(request(undefined, headers))).status).toBe(403); expect(createClient).not.toHaveBeenCalled();
  });

  it.each([
    ["season", "2025-27"], ["team", "GS"], ["opp", "OKC"], ["max_shots", "999"], ["max_shots", "100001"],
    ["max_shots", "1000.1"], ["max_shots", "1e3"], ["version", "not-a-hash"], ["shot_type", "x".repeat(81)],
    ["zone", "x\r\nheader"], ["path", "https://other.example.test"],
  ])("rejects invalid %s before authentication or fetch", async (key, value) => {
    const query = params(); query.set(key, value);
    expect((await GET(request(query.toString()))).status).toBe(400); expect(createClient).not.toHaveBeenCalled();
  });

  it("rejects duplicate and missing parameters, and never builds an unversioned link", async () => {
    const duplicate = params(); duplicate.append("team", "LAL");
    expect((await GET(request(duplicate.toString()))).status).toBe(400);
    const missing = params(); missing.delete("version");
    expect((await GET(request(missing.toString()))).status).toBe(400);
    expect(shotHeatmapExportUrl(inputs, undefined)).toBeNull();
    expect(parseShotHeatmapExport(new URLSearchParams())).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([[401, 401], [403, 403], [404, 404], [422, 422], [429, 429], [503, 503], [500, 502], [302, 502]])("keeps upstream %i out of attachments", async (upstreamStatus, expected) => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: upstreamStatus }));
    const response = await GET(request());
    expect(response.status).toBe(expected); expect(response.headers.has("content-disposition")).toBe(false);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.text()).not.toContain("private-fixture-token");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("requires Render again if the canonical dataset changed after the displayed image", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, dataset_version: `sha256:${"b".repeat(64)}` }));
    const response = await GET(request());
    expect(response.status).toBe(409); expect(response.headers.has("content-disposition")).toBe(false);
    expect(await response.text()).toContain("Render the heatmap again");
  });

  it.each([
    { season: "2024-25" }, { team: "TOR" }, { opp: "BOS" }, { max_shots: 35000 }, { shot_type: null }, { zone: "rim" },
    { dataset_version: undefined }, { n_shots: 1001 }, { n_shots_total: 1000 }, { n_shots_rendered: 1001 },
    { n_shots_omitted_by_cap: -1 }, { n_shots_outside_halfcourt: "2" }, { n_shots_excluded_location: 0.5 },
    { n_shots: 999, n_shots_rendered: 999, n_shots_total: 1082 },
  ])("rejects a response with mismatched filters, source or counts", async change => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, ...change }));
    const response = await GET(request());
    expect(response.status).toBe(502); expect(response.headers.has("content-disposition")).toBe(false);
  });

  it.each(["", "<html>error</html>", Buffer.from("Not a PNG").toString("base64"), png.slice(0, -4), `data:image/png;base64,${png}`])("rejects malformed PNG encoding or contents", async image => {
    fetchMock.mockResolvedValueOnce(Response.json({ ...saved, image_base64: image }));
    const response = await GET(request());
    expect(response.status).toBe(502); expect(response.headers.has("content-disposition")).toBe(false);
  });

  it("rejects excessive dimensions, missing image data, trailing bytes and malformed chunk lengths", async () => {
    const original = Buffer.from(png, "base64");
    const big = Buffer.from(original); big.writeUInt32BE(10000, 16);
    const truncated = Buffer.from(original); truncated.writeUInt32BE(0x7fffffff, 8);
    const noData = Buffer.concat([original.subarray(0, 33), original.subarray(-12)]);
    for (const bytes of [big, truncated, noData, Buffer.concat([original, Buffer.from("extra")])]) {
      fetchMock.mockResolvedValueOnce(Response.json({ ...saved, image_base64: bytes.toString("base64") }));
      const response = await GET(request());
      expect(response.status).toBe(502); expect(response.headers.has("content-disposition")).toBe(false);
    }
  });

  it("rejects wrong content types, invalid JSON and unbounded bodies", async () => {
    for (const upstream of [new Response("<html>error</html>", { headers: { "content-type": "text/html" } }),
      new Response("{", { headers: { "content-type": "application/json" } }),
      Response.json(saved, { headers: { "content-length": String(4 * 1024 * 1024 + 1) } }),
      new Response(" ".repeat(4 * 1024 * 1024 + 1), { headers: { "content-type": "application/json" } })]) {
      fetchMock.mockResolvedValueOnce(upstream);
      const response = await GET(request());
      expect(response.status).toBe(502); expect(response.headers.has("content-disposition")).toBe(false);
    }
  });

  it("does not expose credentials or create a download after a network failure", async () => {
    fetchMock.mockRejectedValueOnce(new Error("private-fixture-token"));
    const response = await GET(request());
    expect(response.status).toBe(502); expect(response.headers.has("content-disposition")).toBe(false);
    expect(await response.text()).not.toContain("private-fixture-token");
  });
});
