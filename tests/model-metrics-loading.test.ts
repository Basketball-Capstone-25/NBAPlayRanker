import { describe, expect, it, vi } from "vitest";
import { LatestRequest } from "../app/infrastructure/latest-request";

function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function view<T>() {
  const state = { loading: false, result: null as T | null, error: null as string | null };
  return { state, handlers: {
    start: () => { state.loading = true; state.error = null; state.result = null; },
    success: (value: T) => { state.result = value; },
    error: (error: unknown) => { state.error = (error as Error).message; },
    settled: () => { state.loading = false; },
  } };
}

describe("model metrics loading", () => {
  it("finishes the pipeline summary and holdout results independently in either response order", async () => {
    for (const first of ["summary", "holdout"]) {
      const summaryRequest = new LatestRequest(), holdoutRequest = new LatestRequest();
      const summary = view<string>(), holdout = view<string>();
      const a = deferred<string>(), b = deferred<string>();
      const doneA = summaryRequest.run("pipeline", () => a.promise, summary.handlers);
      const doneB = holdoutRequest.run("folds=5", () => b.promise, holdout.handlers);
      expect(summary.state.loading && holdout.state.loading).toBe(true);
      if (first === "summary") { a.resolve("pipeline loaded"); await doneA; expect(summary.state.loading).toBe(false); expect(holdout.state.loading).toBe(true); b.resolve("metrics loaded"); }
      else { b.resolve("metrics loaded"); await doneB; expect(holdout.state.loading).toBe(false); expect(summary.state.loading).toBe(true); a.resolve("pipeline loaded"); }
      await Promise.all([doneA, doneB]);
      expect(summary.state).toEqual({ loading: false, result: "pipeline loaded", error: null });
      expect(holdout.state).toEqual({ loading: false, result: "metrics loaded", error: null });
    }
  });

  it("makes one backend call for duplicate initial evaluation requests", async () => {
    const slot = new LatestRequest(), state = view<string>(), pending = deferred<string>();
    const fetch = vi.fn(() => pending.promise);
    const first = slot.run("folds=5", fetch, state.handlers);
    const second = slot.run("folds=5", fetch, state.handlers);
    await Promise.resolve();
    expect(fetch).toHaveBeenCalledOnce(); expect(second).toBe(first);
    pending.resolve("results"); await second;
    expect(state.state).toEqual({ loading: false, result: "results", error: null });
  });

  it("clears loading after failure and supports a successful retry", async () => {
    const slot = new LatestRequest(), state = view<string>();
    await slot.run("folds=5", () => Promise.reject(new Error("Service unavailable")), state.handlers);
    expect(state.state).toEqual({ loading: false, result: null, error: "Service unavailable" });
    await slot.run("folds=5", () => Promise.resolve("recovered"), state.handlers);
    expect(state.state).toEqual({ loading: false, result: "recovered", error: null });
  });

  it("never replaces a new fold selection with a slower previous result or clears its loading early", async () => {
    const slot = new LatestRequest(), state = view<string>(), old = deferred<string>(), current = deferred<string>();
    const oldDone = slot.run("folds=4", () => old.promise, state.handlers);
    const currentDone = slot.run("folds=5", () => current.promise, state.handlers);
    old.resolve("old results"); await oldDone;
    expect(state.state).toEqual({ loading: true, result: null, error: null });
    current.resolve("current results"); await currentDone;
    expect(state.state).toEqual({ loading: false, result: "current results", error: null });
  });
});
