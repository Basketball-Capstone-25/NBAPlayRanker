import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

const { getSession, mockGetUser, mockSingle } = vi.hoisted(() => ({
  getSession: vi.fn(),
  mockGetUser: vi.fn(),
  mockSingle: vi.fn(),
}));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
vi.mock("@supabase/ssr", () => ({
  createServerClient: () => ({
    auth: { getUser: mockGetUser },
    from: () => ({ select: () => ({ eq: () => ({ single: mockSingle }) }) }),
  }),
}));

import { fetchRequestLog, requestLogUrl, RequestLogForbiddenError } from "../app/infrastructure/request-log";
import { middleware } from "../middleware";

describe("Request log client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "analyst-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("sends only the filters that are set, plus paging", () => {
    const url = new URL(requestLogUrl({ user: " coach@ ", endpoint: "", dateFrom: "2026-10-20", page: 2 }));
    expect(url.pathname).toMatch(/\/admin\/request-log$/);
    expect(Object.fromEntries(url.searchParams)).toEqual({ user: "coach@", date_from: "2026-10-20", page: "2", page_size: "50" });
  });

  it("sends the analyst's token and returns the page", async () => {
    const page = { items: [], page: 1, page_size: 50, total: 0, has_next: false };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(page), { status: 200 }));
    await expect(fetchRequestLog({})).resolves.toEqual(page);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer analyst-token" });
  });

  it("treats 403 as not allowed", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "Role 'coach' cannot access" }), { status: 403 }));
    await expect(fetchRequestLog({})).rejects.toBeInstanceOf(RequestLogForbiddenError);
  });

  it("surfaces filter errors with their message", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "The start date must be on or before the end date." }), { status: 400 }));
    await expect(fetchRequestLog({})).rejects.toThrow("The start date must be on or before the end date.");
  });
});

describe("Request log page access", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://test.supabase.co";
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY = "test-key";
  });

  it("lets an analyst open /request-log", async () => {
    mockGetUser.mockResolvedValue({ data: { user: { id: "a1" } } });
    mockSingle.mockResolvedValue({ data: { role: "analyst" } });
    const res = await middleware(new NextRequest("http://localhost:3000/request-log"));
    expect(res.status).toBe(200);
  });

  it("sends a coach away from /request-log", async () => {
    mockGetUser.mockResolvedValue({ data: { user: { id: "c1" } } });
    mockSingle.mockResolvedValue({ data: { role: "coach" } });
    const res = await middleware(new NextRequest("http://localhost:3000/request-log"));
    expect(res.headers.get("location")).toContain("/matchup");
  });
});
