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

describe("Coach access control in middleware", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://test.supabase.co";
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY = "test-key";
  });

  it("redirects signed-in coach from auth page to /matchup", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { id: "c1", user_metadata: { role: "coach" } } },
    });
    mockSingle.mockResolvedValue({ data: { role: "coach" } });

    const req = new NextRequest("http://localhost:3000/login");
    const res = await middleware(req);

    expect(res.status).toBeGreaterThanOrEqual(300);
    expect(res.status).toBeLessThan(400);
    expect(res.headers.get("location")).toContain("/matchup");
  });

  it("allows coach to access /matchup", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { id: "c2", user_metadata: { role: "coach" } } },
    });
    mockSingle.mockResolvedValue({ data: { role: "coach" } });

    const req = new NextRequest("http://localhost:3000/matchup");
    const res = await middleware(req);

    expect(res.status).toBe(200);
    expect(res.headers.get("location")).toBeNull();
  });

  it("redirects coach away from analyst-only /data-explorer", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: { id: "c3", user_metadata: { role: "coach" } } },
    });
    mockSingle.mockResolvedValue({ data: { role: "coach" } });

    const req = new NextRequest("http://localhost:3000/data-explorer");
    const res = await middleware(req);

    expect(res.status).toBeGreaterThanOrEqual(300);
    expect(res.status).toBeLessThan(400);
    expect(res.headers.get("location")).toContain("/matchup");
  });
});
