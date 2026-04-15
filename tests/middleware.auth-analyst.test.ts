import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

const mockGetUser = vi.fn();
const mockSingle = vi.fn();

vi.mock("@supabase/ssr", () => ({
  createServerClient: () => ({
    auth: {
      getUser: mockGetUser,
    },
    from: () => ({
      select: () => ({
        eq: () => ({
          single: mockSingle,
        }),
      }),
    }),
  }),
}));

import { middleware } from "../middleware";

describe("Analyst access control in middleware", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://test.supabase.co";
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY = "test-key";
  });

  it("allows analyst to access /data-explorer", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { id: "u1", user_metadata: { role: "analyst" } } },
    });
    mockSingle.mockResolvedValue({ data: { role: "analyst" } });

    const req = new NextRequest("http://localhost:3000/data-explorer");
    const res = await middleware(req);

    expect(res.status).toBe(200);
    expect(res.headers.get("location")).toBeNull();
  });

  it("redirects coach away from /data-explorer", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { id: "u2", user_metadata: { role: "coach" } } },
    });
    mockSingle.mockResolvedValue({ data: { role: "coach" } });

    const req = new NextRequest("http://localhost:3000/data-explorer");
    const res = await middleware(req);

    expect(res.status).toBeGreaterThanOrEqual(300);
    expect(res.status).toBeLessThan(400);
    expect(res.headers.get("location")).toContain("/matchup");
  });

  it("redirects unauthenticated user to /login with next", async () => {
    mockGetUser.mockResolvedValue({ data: { user: null } });

    const req = new NextRequest("http://localhost:3000/data-explorer?x=1");
    const res = await middleware(req);

    expect(res.status).toBeGreaterThanOrEqual(300);
    expect(res.status).toBeLessThan(400);
    const location = res.headers.get("location") || "";
    expect(location).toContain("/login");
    expect(location).toContain("next=%2Fdata-explorer%3Fx%3D1");
  });
});
