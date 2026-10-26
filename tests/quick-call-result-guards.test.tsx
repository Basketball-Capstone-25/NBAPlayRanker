import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { QuickCallSituation } from "../app/domain/quick-call";
import type { QuickCallResponse } from "../app/infrastructure/quick-call";
import { emptyQuickCallResult, QuickCallRequest, selectQuickCallResult, type QuickCallResultState } from "../app/infrastructure/quick-call-result";

const mocks = vi.hoisted(() => ({ state: null as any, params: new URLSearchParams() }));
vi.mock("next/navigation", () => ({ useSearchParams: () => mocks.params }));
vi.mock("../app/ui/recommendation/_components/useQuickCallResult", () => ({
  useQuickCallResult: (situation: QuickCallSituation) => selectQuickCallResult(mocks.state, situation),
}));
import QuickCallModeClient from "../app/ui/recommendation/_components/QuickCallModeClient";
import { QuickCallForbiddenError } from "../app/services/quick-call";

const base: QuickCallSituation = { season: "2024-25", ourTeam: "TOR", oppTeam: "BOS", margin: -3, period: 4, timeRemaining: 25 };
function response(play = "Spotup"): QuickCallResponse {
  return { source: "context", notice: "", plays: [{ rank: 1, play_type: play, ppp: 1.1, shot_zone: "Corner 3", reason: "Late and trailing." }],
    situation: { season: base.season, our_team: base.ourTeam, opp_team: base.oppTeam, margin: base.margin, period: base.period, time_remaining: base.timeRemaining }, k: 3, elapsed_ms: 12 };
}
function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const publish = (state: QuickCallResultState) => { mocks.state = state; };
function show(situation = base) {
  mocks.params = new URLSearchParams({ season: situation.season, our: situation.ourTeam, opp: situation.oppTeam,
    margin: String(situation.margin), period: String(situation.period), time: String(situation.timeRemaining) });
  return renderToStaticMarkup(<QuickCallModeClient />);
}
beforeEach(() => { mocks.state = emptyQuickCallResult(); });

describe("Quick-Call results follow the displayed situation", () => {
  it("hides the old calls immediately for every changed input, before the debounced request", async () => {
    await new QuickCallRequest().run(base, async () => response(), publish);
    expect(show()).toContain('aria-label="Top 3 plays"');
    const changes = { season: "2025-26", ourTeam: "LAL", oppTeam: "DEN", margin: 2, period: 5, timeRemaining: 15 };
    for (const [key, value] of Object.entries(changes)) {
      const html = show({ ...base, [key]: value });
      expect(html).not.toContain('aria-label="Top 3 plays"');
      expect(html).not.toContain("Context-aware plays");
      expect(html).toContain("Loading the top 3 plays");
    }
  });

  it("clears old calls while refreshing and after a failed refresh, then shows recovery", async () => {
    const request = new QuickCallRequest();
    await request.run(base, async () => response(), publish);
    const pending = deferred<QuickCallResponse>();
    const run = request.run(base, () => pending.promise, publish);
    expect(show()).not.toContain('aria-label="Top 3 plays"');
    pending.reject(new Error("Service unavailable")); await run;
    expect(show()).toContain("Service unavailable");
    expect(show()).not.toContain('aria-label="Top 3 plays"');
    await request.run(base, async () => response("PRRollMan"), publish);
    expect(show()).toContain("PRRollMan");
    expect(show()).not.toContain("Service unavailable");
  });

  it("cancels the old response during the debounce without depending on fetch honoring abort", async () => {
    const request = new QuickCallRequest(), old = deferred<QuickCallResponse>();
    let signal: AbortSignal | undefined;
    const run = request.run(base, (_snapshot, nextSignal) => { signal = nextSignal; return old.promise; }, publish);
    request.cancel(); // the control-change cleanup runs before the next request starts
    expect(signal?.aborted).toBe(true);
    old.resolve(response("Old play")); await run;
    expect(mocks.state.result).toBeNull();
  });

  it("ignores an older completion and failure after a newer result succeeds", async () => {
    for (const fail of [false, true]) {
      const request = new QuickCallRequest(), old = deferred<QuickCallResponse>();
      const oldRun = request.run(base, () => old.promise, publish);
      const next = { ...base, margin: 2 };
      await request.run(next, async () => response("Latest play"), publish);
      if (fail) old.reject(new Error("Old request failed")); else old.resolve(response("Old play"));
      await oldRun;
      expect(selectQuickCallResult(mocks.state, next).result?.plays[0].play_type).toBe("Latest play");
      expect(show(next)).toContain("Latest play");
      expect(show(next)).not.toContain("Old request failed");
    }
  });

  it("retains the successful request snapshot and preserves coach-only denial", async () => {
    const request = new QuickCallRequest(), inputs = { ...base };
    await request.run(inputs, async () => response(), publish);
    inputs.margin = 10;
    expect(mocks.state.situation.margin).toBe(-3);
    await request.run(base, async () => { throw new QuickCallForbiddenError("Coach access required"); }, publish);
    expect(show()).toContain("Coach access required");
    expect(show()).not.toContain('aria-label="Top 3 plays"');
  });
});
