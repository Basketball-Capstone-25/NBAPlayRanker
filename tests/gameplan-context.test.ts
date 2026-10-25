import { afterEach, describe, expect, it, vi } from "vitest";
import { periodLabel, resolveBuildSituation, resolveGameSituation } from "../app/domain/gameplan-context";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: async () => ({ data: { session: { access_token: "test-session" } } }) } }) }));
import { contextRank } from "../app/infrastructure/api-client";

const initial = resolveGameSituation({}, { period: 4, time_remaining: 180, margin: -2 });
afterEach(() => vi.unstubAllGlobals());

describe("Gameplan situation used by Build", () => {
  it("sends the new situation on the first request and the same context on a second Build", async () => {
    const fetchMock = vi.fn(async (_url: string, _options?: RequestInit) => new Response(JSON.stringify({ rankings: [] })));
    vi.stubGlobal("fetch", fetchMock);
    const parse = vi.fn(async () => ({ period: 4, time_remaining: 28, margin: -5, need3: true, vs_switching: true }));
    const first = await resolveBuildSituation("Q4 down 5 with 0:28 left. Need a three. They are switching.", initial, parse);
    const second = await resolveBuildSituation("Q4 down 5 with 0:28 left. Need a three. They are switching.", first, parse);
    for (const situation of [first, second]) await contextRank({ season: "2024-25", our: "TOR", opp: "BOS", margin: situation.margin, period: situation.period, timeRemaining: situation.time_remaining, coaching: situation.coaching });
    expect(first).toEqual(second);
    const urls = fetchMock.mock.calls.map(call => new URL(String(call[0])));
    expect(urls[0].search).toBe(urls[1].search);
    expect(urls[0].searchParams.get("margin")).toBe("-5");
    expect(urls[0].searchParams.get("time_remaining")).toBe("28");
    expect(JSON.parse(urls[0].searchParams.get("coaching")!)).toMatchObject({ need3: true, vs_switching: true });
    expect(String(urls[0])).not.toContain("test-session");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ headers: { Authorization: "Bearer test-session" } });
  });

  it("keeps OT from parser through the request and display", async () => {
    const fetchMock = vi.fn(async (_url: string, _options?: RequestInit) => new Response(JSON.stringify({ rankings: [] })));
    vi.stubGlobal("fetch", fetchMock);
    const situation = await resolveBuildSituation("OT down 2 with 0:28 left", initial, async () => ({ period: 5, time_remaining: 28, margin: -2 }));
    await contextRank({ season: "2024-25", our: "TOR", opp: "BOS", margin: situation.margin, period: situation.period, timeRemaining: situation.time_remaining, coaching: situation.coaching });
    const url = new URL(String(fetchMock.mock.calls[0][0]));
    expect(url.searchParams.get("period")).toBe("5");
    expect(url.searchParams.get("time_remaining")).toBe("28");
    expect(periodLabel(situation.period)).toBe("OT");
  });

  it("limits an OT clock to five minutes without changing the period", () => {
    const situation = resolveGameSituation({ period: 5, time_remaining: 600 }, initial);
    expect(situation.period).toBe(5);
    expect(situation.time_remaining).toBe(300);
    expect(situation.warnings.join(" ")).toContain("Clock adjusted");
  });

  it("lists unsupported conditions and does not send them as ranking inputs", () => {
    const situation = resolveGameSituation({ need: "foul_game", foul_game: true, defense_style: "man", pace: "balanced", end_of_quarter: true }, initial);
    expect(situation.coaching).toEqual({});
    expect(situation.warnings.join(" ")).toContain("does not change the ranking");
    expect(situation.warnings.join(" ")).toContain("End of period is a note");
  });

  it("does not rank old values when parsing fails", async () => {
    const parse = vi.fn(async () => { throw new Error("Parser unavailable"); });
    await expect(resolveBuildSituation("new situation", initial, parse)).rejects.toThrow("Parser unavailable");
  });

  it("keeps numeric controls when there is no prompt", async () => {
    const parse = vi.fn();
    expect(await resolveBuildSituation("", initial, parse)).toEqual(initial);
    expect(parse).not.toHaveBeenCalled();
  });
});
