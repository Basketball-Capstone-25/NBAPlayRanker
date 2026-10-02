import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockSignIn, mockSignUp, mockProfile } = vi.hoisted(() => ({
  mockSignIn: vi.fn(), mockSignUp: vi.fn(), mockProfile: vi.fn(),
}));

vi.mock("../lib/supabase/client", () => ({
  createClient: () => ({
    auth: { signInWithPassword: mockSignIn, signUp: mockSignUp },
    from: () => ({ select: () => ({ eq: () => ({ maybeSingle: mockProfile }) }) }),
  }),
}));

import { signIn, signUp } from "../app/infrastructure/auth";

describe("administrator-assigned profile roles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSignIn.mockResolvedValue({ data: { user: { id: "u1", user_metadata: { role: "coach" } } } });
  });

  it("uses the profile role despite edited signup metadata", async () => {
    mockProfile.mockResolvedValue({ data: { role: "analyst" } });
    await expect(signIn("test@example.test", "password")).resolves.toEqual({ userId: "u1", role: "analyst" });
  });

  it("shows pending access instead of granting an unassigned account a role", async () => {
    mockProfile.mockResolvedValue({ data: { role: null } });
    await expect(signIn("test@example.test", "password")).rejects.toThrow("awaiting role approval");
  });

  it("submits a requested role rather than an authorization claim", async () => {
    process.env.NEXT_PUBLIC_SITE_URL = "https://example.test";
    mockSignUp.mockResolvedValue({ error: null });
    await signUp("test@example.test", "password", "coach");
    expect(mockSignUp).toHaveBeenCalledWith(expect.objectContaining({
      options: expect.objectContaining({ data: { requested_role: "coach" } }),
    }));
  });
});
