import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

import {
  LATE_GAME_SCENARIOS,
  applyScenario,
  canStep,
  clampSituation,
  formatClock,
  formatMargin,
  formatPeriod,
  quickCallHref,
  situationFromParams,
  stepSituation,
  type QuickCallSituation,
} from "../app/domain/quick-call";
import { fetchQuickCall, quickCallUrl, QuickCallForbiddenError } from "../app/infrastructure/quick-call";

const base: QuickCallSituation = { season: "2024-25", ourTeam: "TOR", oppTeam: "BOS", margin: -2, period: 4, timeRemaining: 40 };

describe("Quick-Call situation rules", () => {
  it("moves score and period by one and the clock by five seconds", () => {
    expect(stepSituation(base, "margin", 1).margin).toBe(-1);
    expect(stepSituation(base, "period", -1).period).toBe(3);
    expect(stepSituation(base, "timeRemaining", -1).timeRemaining).toBe(35);
  });

  it("keeps values on a real scoreboard and disables buttons at the limits", () => {
    expect(clampSituation({ ...base, margin: 55 }).margin).toBe(40);
    expect(clampSituation({ ...base, period: 5, timeRemaining: 700 }).timeRemaining).toBe(300);
    expect(canStep({ ...base, timeRemaining: 0 }, "timeRemaining", -1)).toBe(false);
    expect(canStep({ ...base, period: 5 }, "period", 1)).toBe(false);
  });

  it("loads a preloaded late-game scenario in one tap", () => {
    const down3 = LATE_GAME_SCENARIOS.find((s) => s.id === "down3-under30")!;
    expect(applyScenario(base, down3)).toEqual({ ...base, margin: -3, period: 4, timeRemaining: 25 });
    expect(LATE_GAME_SCENARIOS.every((s) => clampSituation({ ...base, ...s }).timeRemaining === s.timeRemaining)).toBe(true);
  });

  it("round-trips the situation through the Context ML link", () => {
    const href = quickCallHref(base);
    expect(href.startsWith("/context/quick-call?")).toBe(true);
    expect(situationFromParams(new URL(href, "https://app.test").searchParams)).toEqual(base);
  });

  it("formats the scoreboard for large text", () => {
    expect(formatClock(25)).toBe("0:25");
    expect(formatClock(605)).toBe("10:05");
    expect(formatPeriod(5)).toBe("OT");
    expect(formatMargin(-3)).toBe("Down 3");
    expect(formatMargin(0)).toBe("Tied");
  });
});

describe("Quick-Call API client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "coach-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("sends the situation and the coach's token", async () => {
    const body = { source: "context", notice: "", plays: [], situation: {}, k: 3, elapsed_ms: 12 };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }));

    await expect(fetchQuickCall(base)).resolves.toEqual(body);

    const url = new URL(fetchMock.mock.calls[0][0]);
    expect(url.pathname).toMatch(/\/rank-plays\/quick-call$/);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      season: "2024-25", our: "TOR", opp: "BOS", margin: "-2", period: "4", time_remaining: "40",
    });
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer coach-token" });
    expect(quickCallUrl(base)).toBe(fetchMock.mock.calls[0][0]);
  });

  it("treats 403 as not allowed rather than a failure", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "Role 'analyst' cannot access" }), { status: 403 }));
    await expect(fetchQuickCall(base)).rejects.toBeInstanceOf(QuickCallForbiddenError);
  });

  it("surfaces other API errors with their message", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "Score margin must be between -40 and 40." }), { status: 400 }));
    await expect(fetchQuickCall(base)).rejects.toThrow("Score margin must be between -40 and 40.");
  });
});
