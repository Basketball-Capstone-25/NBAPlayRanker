import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getUser, getSession, serverClient, cookies, cookieStore } = vi.hoisted(() => ({
  getUser: vi.fn(), getSession: vi.fn(), serverClient: vi.fn(), cookies: vi.fn(),
  cookieStore: { getAll: vi.fn(), set: vi.fn() },
}));
vi.mock("@supabase/ssr", () => ({ createServerClient: serverClient }));
vi.mock("next/headers", () => ({ cookies }));
import { GET } from "../app/api/exports/route";

describe("same-origin export proxy", () => {
  const fetchMock = vi.fn();
  function request(path = "/data/team-playtypes.csv", query = "season=2024-25&team=TOR", extra: Record<string, string> = {}, headers: HeadersInit = {}) {
    return new Request(`https://frontend.example.test/api/exports?${new URLSearchParams({ path, query, ...extra })}`, { headers });
  }
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NEXT_PUBLIC_API_BASE", "https://api.example.test");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://auth.example.test");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "public-test-key");
    vi.stubGlobal("fetch", fetchMock);
    cookieStore.getAll.mockReturnValue([{ name: "sb-session", value: "cookie-fixture" }]);
    cookies.mockResolvedValue(cookieStore);
    serverClient.mockReturnValue({ auth: { getUser, getSession } });
    getUser.mockResolvedValue({ data: { user: { id: "analyst-user" } }, error: null });
    getSession.mockResolvedValue({ data: { session: { access_token: "cookie-session-token", user: { id: "analyst-user" } } }, error: null });
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

  it("verifies the SSR cookie identity, forwards its token and streams the exact CSV", async () => {
    const csv = "PLAY_TYPE,PPP\nCut,1.2\n";
    const upstream = new Response(csv, { headers: { "content-type": "text/csv; charset=utf-8" } });
    fetchMock.mockResolvedValue(upstream);
    const query = "season=2024-25&team=TOR&play_type=Cut%20%26%20Roll&tag=one&tag=two";
    const response = await GET(request("/data/team-playtypes.csv", query, { filename: "TOR.csv" }, { Authorization: "Bearer ignored-browser-header" }));

    expect(cookies).toHaveBeenCalledOnce();
    expect(serverClient.mock.calls[0][2].cookies.getAll()).toEqual([{ name: "sb-session", value: "cookie-fixture" }]);
    expect(getUser).toHaveBeenCalledOnce();
    expect(getSession).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toBe(`https://api.example.test/data/team-playtypes.csv?${query}`);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ headers: { Authorization: "Bearer cookie-session-token" }, cache: "no-store", redirect: "manual" });
    expect(response.status).toBe(200);
    expect(response.body).toBe(upstream.body);
    expect(response.headers.get("content-disposition")).toBe('attachment; filename="TOR.csv"');
    expect(response.headers.get("content-type")).toBe("text/csv; charset=utf-8");
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(await response.text()).toBe(csv);
  });

  it("returns 401 without attachment or upstream fetch for an invalid cookie session", async () => {
    getUser.mockResolvedValue({ data: { user: null }, error: { message: "invalid session" } });
    const response = await GET(request());
    expect(response.status).toBe(401);
    expect(response.headers.has("content-disposition")).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a cookie session that disagrees with the verified identity", async () => {
    getSession.mockResolvedValue({ data: { session: { access_token: "different-token", user: { id: "other-user" } } }, error: null });
    expect((await GET(request())).status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 403, 422])("preserves upstream denial %s as JSON, never an attachment", async status => {
    fetchMock.mockResolvedValue(new Response('{"detail":"Denied"}', { status }));
    const response = await GET(request());
    expect(response.status).toBe(status);
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(response.headers.has("content-disposition")).toBe(false);
    expect(await response.text()).not.toContain("cookie-session-token");
  });

  it.each(["https://foreign.example.test/file.csv", "//foreign.example.test/file.csv", "/meta/options", "/data/../meta/options"])(
    "rejects unapproved paths without authentication or upstream fetch: %s", async path => {
      const response = await GET(request(path));
      expect(response.status).toBe(400);
      expect(serverClient).not.toHaveBeenCalled();
      expect(fetchMock).not.toHaveBeenCalled();
    }
  );

  it.each([{ Origin: "https://foreign.example.test" }, { "Sec-Fetch-Site": "cross-site" }])(
    "rejects cross-origin requests", async headers => {
      expect((await GET(request(undefined, undefined, {}, headers))).status).toBe(403);
      expect(fetchMock).not.toHaveBeenCalled();
    }
  );

  it("rejects arbitrary target parameters and duplicate route selectors", async () => {
    expect((await GET(request(undefined, undefined, { url: "https://foreign.example.test" }))).status).toBe(400);
    const duplicate = new Request(request().url + "&path=/export/shotplan.pdf");
    expect((await GET(duplicate)).status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("preserves PDF headers and a safe server filename", async () => {
    fetchMock.mockResolvedValue(new Response("%PDF-fixture", { headers: { "content-type": "application/pdf", "content-disposition": 'attachment; filename="TOR_shotplan.pdf"' } }));
    const response = await GET(request("/export/shotplan.pdf"));
    expect(response.headers.get("content-disposition")).toBe('attachment; filename="TOR_shotplan.pdf"');
    expect(await response.text()).toBe("%PDF-fixture");
  });

  it("supplies the route filename and strips unsafe characters in requested names", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(new Response("a,b\n1,2", { headers: { "content-type": "text/csv" } })));
    const fallback = await GET(request("/rank-plays/baseline.csv"));
    expect(fallback.headers.get("content-disposition")).toBe('attachment; filename="baseline.csv"');
    const safe = await GET(request(undefined, undefined, { filename: "../bad\r\nname.exe" }));
    expect(safe.headers.get("content-disposition")).toBe('attachment; filename="_bad__name.exe.csv"');
  });

  it("does not follow redirects or serve a successful JSON error as a CSV", async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 302, headers: { Location: "https://foreign.example.test" } }));
    const redirect = await GET(request());
    expect(redirect.status).toBe(502);
    expect(redirect.headers.has("content-disposition")).toBe(false);
    fetchMock.mockResolvedValueOnce(Response.json({ error: "unexpected payload" }));
    const wrongType = await GET(request());
    expect(wrongType.status).toBe(502);
    expect(wrongType.headers.has("content-disposition")).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
