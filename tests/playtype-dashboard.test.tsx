import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import example from "./fixtures/playtype-dashboard.json";
import { dashboardKey, parsePlaytypeDashboard, type DashboardFilters, type PlaytypeDashboard } from "../app/domain/playtype-dashboard";
import { DashboardRequest, emptyDashboard, fetchPlaytypeDashboard, selectDashboard, withRequestDeadline, type DashboardState } from "../app/infrastructure/playtype-dashboard";
import { DashboardResults, DashboardResultState } from "../app/ui/recommendation/_components/MatchupDashboardClient";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));
const filters: DashboardFilters = { season: "2024-25", our: "TOR", opp: "BOS", min_poss: 20 };
const report = () => structuredClone(example) as PlaytypeDashboard;
const display = (data = report(), view: "usage" | "defense" = "usage") => renderToStaticMarkup(<DashboardResults data={data} view={view} />);
const deferred = <T,>() => {
  let resolve!: (value: T) => void, reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

describe("play-type comparison contract", () => {
  it("accepts independently aggregated regular-season totals and keeps unrounded values", () => {
    const data = parsePlaytypeDashboard(report(), filters);
    expect(data.usage_efficiency.total_possessions).toBe(8517);
    expect(data.usage_efficiency.rows.find(r => r.play_type === "Cut")?.ppp).toBe(892 / 745);
    expect(data.opponent_defense.rows[0].play_type).toBe("Handoff");
  });
  it.each(["season", "our", "opp", "min_poss"] as const)("rejects results for an older %s selection", key => {
    const next = { ...filters, [key]: key === "min_poss" ? 100 : "different" };
    expect(() => parsePlaytypeDashboard(report(), next)).toThrow("incomplete");
  });
  it.each([
    ["playoff scope", (v: any) => { v.filters.season_phase = "playoffs"; }],
    ["missing source", (v: any) => { v.dataset_versions = {}; }],
    ["duplicate play", (v: any) => { v.usage_efficiency.rows[1].play_type = "Cut"; }],
    ["wrong possession sum", (v: any) => { v.usage_efficiency.total_possessions++; }],
    ["wrong PPP", (v: any) => { v.usage_efficiency.rows[0].ppp += .2; }],
    ["wrong usage denominator", (v: any) => { v.usage_efficiency.rows[0].usage_pct = 50; }],
    ["non-finite efficiency", (v: any) => { v.usage_efficiency.rows[0].ppp = NaN; }],
    ["false sample flag", (v: any) => { v.usage_efficiency.rows[0].sample_ok = false; }],
    ["unavailable zero row", (v: any) => { v.coverage.unavailable_defense_types.push("Handoff"); }],
    ["inverted defensive order", (v: any) => { v.opponent_defense.rows.reverse(); }],
    ["false weak spot", (v: any) => { v.opponent_defense.rows[6].weak_spot_rank = 3; }],
    ["missing top weak spot", (v: any) => { v.opponent_defense.rows[0].weak_spot_rank = null; }],
    ["wrong defensive total", (v: any) => { v.opponent_defense.tracked_possessions++; }],
  ])("rejects %s before drawing a chart", (_name, change) => {
    const data = report(); (change as (v: any) => void)(data);
    expect(() => parsePlaytypeDashboard(data, filters)).toThrow("incomplete");
  });
  it("retains all low-sample rows when no weak spot qualifies", () => {
    const data = report(); data.filters.min_poss = 5000;
    data.usage_efficiency.rows.forEach(r => { r.sample_ok = false; });
    data.opponent_defense.rows.forEach(r => { r.sample_ok = false; r.weak_spot_rank = null; });
    expect(parsePlaytypeDashboard(data, { ...filters, min_poss: 5000 }).opponent_defense.rows).toHaveLength(7);
    const html = display(data, "defense");
    expect(html).toContain("No above-league defensive weak spot meets this possession cutoff");
    expect(html).toContain("2,939");
    expect(html).toContain("Small sample");
    expect(html).not.toContain('aria-label="Supported defensive weak spots"');
  });
  it("does not turn positive floating-point noise into a weak spot", () => {
    const data = report();
    data.opponent_defense.rows.forEach((row, i) => {
      row.allowed_ppp_delta = i === 0 ? 5e-13 : 0;
      row.league_allowed_ppp = row.allowed_ppp - row.allowed_ppp_delta;
      row.weak_spot_rank = null;
    });
    expect(parsePlaytypeDashboard(data, filters).opponent_defense.rows[0].weak_spot_rank).toBeNull();
  });
});

describe("comparison requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks(); vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("ends a hung team-list request and permits a fresh retry", async () => {
    vi.useFakeTimers();
    try {
      const old = deferred<string[]>();
      const pending = withRequestDeadline(old.promise, "Team choices took too long");
      const rejection = expect(pending).rejects.toThrow("took too long");
      await vi.advanceTimersByTimeAsync(60000); await rejection;
      await expect(withRequestDeadline(Promise.resolve(["TOR", "BOS"]), "timeout")).resolves.toEqual(["TOR", "BOS"]);
      old.resolve(["OLD"]);
      await expect(pending).rejects.toThrow("took too long");
      expect(vi.getTimerCount()).toBe(0);
    } finally { vi.useRealTimers(); }
  });
  it("sends a bearer header, current filters and abort signal without caching", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report())));
    const signal = new AbortController().signal;
    await expect(fetchPlaytypeDashboard(filters, signal)).resolves.toEqual(report());
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain("/analytics/playtypes?season=2024-25&our=TOR&opp=BOS&min_poss=20");
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ cache: "no-store", signal, headers: { Authorization: "Bearer test-token" } });
  });
  it.each([[401, "Sign in again"], [403, "coach account"], [404, "No regular-season data"], [422, "Check the teams"], [503, "unavailable"]])("handles %s without exposing server details", async (status, message) => {
    fetchMock.mockResolvedValue(new Response("/private/server/trace", { status: Number(status) }));
    await expect(fetchPlaytypeDashboard(filters)).rejects.toThrow(String(message));
    await expect(fetchPlaytypeDashboard(filters)).rejects.not.toThrow("/private/");
  });
  it("handles a connection failure and a malformed successful response", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch")).mockResolvedValueOnce(new Response("not json"));
    await expect(fetchPlaytypeDashboard(filters)).rejects.toThrow("Could not reach");
    await expect(fetchPlaytypeDashboard(filters)).rejects.toThrow("incomplete");
  });
});

describe("current-input results", () => {
  let state: DashboardState;
  const publish = (next: DashboardState) => { state = next; };
  const show = (f = filters) => renderToStaticMarkup(<DashboardResultState state={selectDashboard(state, f)} view="usage" retry={() => {}} />);
  beforeEach(() => { state = emptyDashboard(); });
  it("hides old charts on the render immediately after any filter change", async () => {
    await new DashboardRequest().run(filters, async () => report(), publish);
    expect(show()).toContain("8,517");
    for (const [key, value] of Object.entries({ season: "2023-24", our: "LAL", opp: "DEN", min_poss: 50 })) {
      expect(show({ ...filters, [key]: value })).toContain("Loading the selected teams");
      expect(show({ ...filters, [key]: value })).not.toContain("8,517");
    }
  });
  it("clears old results on refresh, failure and retry", async () => {
    const request = new DashboardRequest();
    await request.run(filters, async () => report(), publish);
    const pending = deferred<PlaytypeDashboard>();
    const run = request.run(filters, () => pending.promise, publish);
    expect(show()).not.toContain("8,517");
    pending.reject(new Error("Service unavailable")); await run;
    expect(show()).toContain("Try again"); expect(show()).not.toContain("8,517");
    await request.run(filters, async () => report(), publish);
    expect(show()).toContain("8,517"); expect(show()).not.toContain("Service unavailable");
  });
  it("ignores an older completion or failure even when fetch does not honor cancellation", async () => {
    for (const fail of [false, true]) {
      const request = new DashboardRequest(), old = deferred<PlaytypeDashboard>();
      const run = request.run(filters, () => old.promise, publish);
      const next = { ...filters, min_poss: 50 };
      await request.run(next, async () => report(), publish);
      if (fail) old.reject(new Error("Old failure")); else old.resolve(report());
      await run;
      expect(state.key).toBe(dashboardKey(next)); expect(state.error).toBeNull();
    }
  });
  it("times out with a retry and cannot show a late result", async () => {
    vi.useFakeTimers();
    try {
      const request = new DashboardRequest(), pending = deferred<PlaytypeDashboard>();
      const run = request.run(filters, () => pending.promise, publish);
      await vi.advanceTimersByTimeAsync(60000);
      expect(show()).toContain("took too long");
      pending.resolve(report()); await run;
      expect(show()).not.toContain("8,517");
    } finally { vi.useRealTimers(); }
  });
});

describe("dashboard evidence displayed to a coach", () => {
  it("shows exact table values, numbered chart, scope and denominators", () => {
    const html = display();
    expect(html).toContain("8.7%"); expect(html).toContain("1.197"); expect(html).toContain("1.307");
    expect(html).toContain("Cut"); expect(html).toContain("League usage");
    expect(html).toContain("regular season"); expect(html).toContain("percentage points");
    expect(html).toContain("not proof that using the play more will improve results");
    expect(html).toContain('role="img"'); expect(html).toContain('<th scope="row">');
  });
  it("shows supported positive weak spots and never invents missing defensive categories", () => {
    const html = display(report(), "defense");
    expect(html).toContain("Area to review 3"); expect(html).not.toContain("Area to review 4");
    expect(html).toContain("+0.073"); expect(html).toContain("1.010");
    expect(html).toContain("Cut, Other, Offensive rebound, Transition");
    expect(html).toContain("missing from the source, not zero");
    expect(html).toContain("same season, side and play type");
  });
});
