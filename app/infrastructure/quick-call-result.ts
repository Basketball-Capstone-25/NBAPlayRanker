import type { QuickCallSituation } from "../domain/quick-call";
import type { QuickCallResponse } from "./quick-call";

export type QuickCallResultState = {
  situation: QuickCallSituation | null;
  result: QuickCallResponse | null;
  loading: boolean;
  error: unknown;
  lastRefreshMs: number | null;
};

export function emptyQuickCallResult(): QuickCallResultState {
  return { situation: null, result: null, loading: true, error: null, lastRefreshMs: null };
}

export function selectQuickCallResult(state: QuickCallResultState, situation: QuickCallSituation) {
  const matches = state.situation !== null &&
    (Object.keys(situation) as (keyof QuickCallSituation)[]).every(key => state.situation![key] === situation[key]);
  // Hide stale calls during the render before the debounced refresh starts.
  return matches ? state : emptyQuickCallResult();
}

export class QuickCallRequest {
  private controller: AbortController | null = null;
  private version = 0;

  cancel() {
    this.version += 1;
    this.controller?.abort();
  }

  async run(
    situation: QuickCallSituation,
    fetchResult: (snapshot: QuickCallSituation, signal: AbortSignal) => Promise<QuickCallResponse>,
    publish: (state: QuickCallResultState) => void,
  ) {
    this.cancel();
    const version = this.version;
    const controller = new AbortController();
    this.controller = controller;
    const snapshot = { ...situation };
    const state = { ...emptyQuickCallResult(), situation: snapshot };
    publish(state);
    const started = performance.now();
    try {
      const result = await fetchResult(snapshot, controller.signal);
      if (version === this.version && !controller.signal.aborted) {
        publish({ ...state, result, loading: false, lastRefreshMs: Math.round(performance.now() - started) });
      }
    } catch (error) {
      if (version === this.version && !controller.signal.aborted) publish({ ...state, loading: false, error });
    }
  }
}
