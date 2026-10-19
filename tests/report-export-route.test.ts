import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
const { getUser, getSession, createClient } = vi.hoisted(() => ({ getUser: vi.fn(), getSession: vi.fn(), createClient: vi.fn() }));
vi.mock("../lib/supabase/server", () => ({ createClient }));
import { POST } from "../app/api/exports/report/route";

const origin = "https://frontend.example.test";
const payload = { report: "baseline_rankings", filters: { season: "2024-25", our: "TOR", opp: "BOS" }, columns: ["Play type", "PPP"], rows: [{ "Play type": "Cut", PPP: 1.2 }], metadata: {} };
function request(format = "csv", form = false, headers: Record<string, string> = {}, value: unknown = { format, payload }) {
  const json = JSON.stringify(value);
  return new Request(`${origin}/api/exports/report`, { method: "POST", headers: { Origin: origin, "Content-Type": form ? "application/x-www-form-urlencoded" : "application/json", ...headers }, body: form ? new URLSearchParams({ request: json }) : json });
}
function file(format = "csv", content?: string) {
  const contentType = { csv: "text/csv", json: "application/json", pdf: "application/pdf" }[format];
  return new Response(content ?? (format === "pdf" ? "%PDF-fixture" : format === "json" ? JSON.stringify({ contract_version: "1", report: payload.report, rows: payload.rows }) : "Play type,PPP\nCut,1.2\n"),
    { headers: { "Content-Type": contentType!, "Content-Disposition": `attachment; filename="baseline-TOR-BOS.${format}"` } });
}

describe("displayed report proxy", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks(); vi.stubEnv("NEXT_PUBLIC_API_BASE", "https://api.example.test"); vi.stubGlobal("fetch", fetchMock);
    createClient.mockResolvedValue({ auth: { getUser, getSession } });
    getUser.mockResolvedValue({ data: { user: { id: "coach" } }, error: null });
    getSession.mockResolvedValue({ data: { session: { user: { id: "coach" }, access_token: "private-token" } }, error: null });
    fetchMock.mockImplementation(() => Promise.resolve(file()));
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

  it("validates the cookie identity and checks the exact generated file before reporting ready", async () => {
    const response = await POST(request());
    expect(response.status).toBe(200); expect(await response.json()).toEqual({ ready: true });
    expect(getUser).toHaveBeenCalledOnce(); expect(getSession).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toBe("https://api.example.test/export/report");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST", headers: { Authorization: "Bearer private-token", "Content-Type": "application/json" }, body: JSON.stringify({ format: "csv", payload }), redirect: "manual", cache: "no-store" });
    expect(response.headers.get("content-disposition")).toBeNull();
    expect(response.headers.get("cache-control")).toBe("private, no-store");
  });

  it.each(["csv", "json", "pdf"])("delivers native form %s downloads with the matching filename and type", async format => {
    fetchMock.mockResolvedValue(file(format));
    const response = await POST(request(format, true));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-disposition")).toBe(`attachment; filename="baseline-TOR-BOS.${format}"`);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(await response.text()).not.toContain("private-token");
  });

  it.each([undefined, "https://foreign.example", "null"])("rejects an absent or foreign Origin (%s)", async originHeader => {
    const req = request(); if (originHeader === undefined) req.headers.delete("origin"); else req.headers.set("origin", originHeader);
    expect((await POST(req)).status).toBe(403); expect(createClient).not.toHaveBeenCalled(); expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects cross-site form submissions and arbitrary query selectors", async () => {
    expect((await POST(request("csv", true, { "Sec-Fetch-Site": "cross-site" }))).status).toBe(403);
    const response = await POST(new Request(`${origin}/api/exports/report?url=https://foreign.test`, { method: "POST", headers: { Origin: origin } }));
    expect(response.status).toBe(400); expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([null, {}, { format: "xlsx", payload }, { format: "csv", payload: { ...payload, report: "other" } }, { format: "csv", payload: { ...payload, rows: [] } }])("rejects unsupported or empty reports", async value => {
    expect((await POST(request("csv", false, {}, value))).status).toBe(400); expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects oversized requests before authentication or backend work", async () => {
    const response = await POST(request("csv", false, {}, { format: "csv", payload: { ...payload, metadata: { huge: "x".repeat(270000) } } }));
    expect(response.status).toBe(413); expect(createClient).not.toHaveBeenCalled(); expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects duplicate native form fields", async () => {
    const req = new Request(`${origin}/api/exports/report`, { method: "POST", headers: { Origin: origin, "Content-Type": "application/x-www-form-urlencoded" }, body: "request=%7B%7D&request=%7B%7D" });
    expect((await POST(req)).status).toBe(400); expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects invalid and mismatched sessions", async () => {
    getUser.mockResolvedValueOnce({ data: { user: null }, error: null });
    expect((await POST(request())).status).toBe(401);
    getSession.mockResolvedValueOnce({ data: { session: { user: { id: "other" }, access_token: "bad-token" } }, error: null });
    expect((await POST(request())).status).toBe(401); expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 403, 422, 503, 302])("never turns upstream HTTP %s into a downloadable error file", async status => {
    fetchMock.mockResolvedValue(new Response("private server detail", { status }));
    const response = await POST(request("csv", true));
    expect(response.status).toBe(status >= 400 && status < 500 ? status : 502);
    expect(response.headers.get("content-disposition")).toBeNull();
    expect(await response.text()).not.toContain("private server detail");
  });

  it("rejects wrong media types, extensions and invalid report signatures", async () => {
    fetchMock.mockResolvedValueOnce(file("json"));
    expect((await POST(request("csv", true))).status).toBe(502);
    fetchMock.mockResolvedValueOnce(new Response("x", { headers: { "Content-Type": "text/csv", "Content-Disposition": 'attachment; filename="x.pdf"' } }));
    expect((await POST(request("csv", true))).status).toBe(502);
    fetchMock.mockResolvedValueOnce(file("pdf", "not a pdf"));
    expect((await POST(request("pdf", true))).status).toBe(502);
    fetchMock.mockResolvedValueOnce(file("json", '{"error":"broken"}'));
    expect((await POST(request("json", true))).status).toBe(502);
  });

  it("preserves the report name and suffix when the server supplies a long bounded filename", async () => {
    const filename = `baseline-rankings_${"x".repeat(162)}.csv`;
    fetchMock.mockResolvedValue(new Response("Play type,PPP\nCut,1.2\n", { headers: {
      "Content-Type": "text/csv", "Content-Disposition": `attachment; filename="${filename}"`,
    } }));
    const response = await POST(request("csv", true));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-disposition")).toBe(`attachment; filename="${filename}"`);
  });
});
