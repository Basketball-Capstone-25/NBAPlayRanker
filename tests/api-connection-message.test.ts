import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

import { postJson, SERVER_UNREACHABLE } from "../app/infrastructure/api-client";

describe("lost server connection message (SCRUM-887)", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-session" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("replaces the browser's Failed to fetch with a plain sentence", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await postJson("https://api.test/x", {}).catch((e) => e);
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe(SERVER_UNREACHABLE);
  });

  it("lets a cancelled request stay cancelled", async () => {
    fetchMock.mockRejectedValue(new DOMException("aborted", "AbortError"));
    await expect(postJson("https://api.test/x", {})).rejects.toMatchObject({ name: "AbortError" });
  });
});
