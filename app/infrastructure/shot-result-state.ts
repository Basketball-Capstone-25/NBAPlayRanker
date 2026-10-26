export type ShotRequestState<I, R> = {
  key: string | null;
  inputs: I | null;
  result: R | null;
  loading: boolean;
  error: string | null;
};

export function emptyShotResult<I, R>(): ShotRequestState<I, R> {
  return { key: null, inputs: null, result: null, loading: false, error: null };
}

export function shotInputKey(inputs: object): string {
  return JSON.stringify(Object.entries(inputs).sort(([a], [b]) => a.localeCompare(b)));
}

/** Only the inputs that produced a successful result may label or export it. */
export function selectShotResult<I extends object, R>(state: ShotRequestState<I, R>, inputs: I) {
  const changed = state.key !== null && state.key !== shotInputKey(inputs);
  const ready = !changed && !state.loading && !state.error && state.result !== null;
  return {
    changed,
    loading: state.loading,
    error: changed ? null : state.error,
    result: ready ? state.result : null,
    inputs: ready ? state.inputs : null,
  };
}

/** Superseded requests cannot replace a newer result or clear its loading state. */
export class ShotResultRequest<I extends object, R> {
  private version = 0;

  cancel() { this.version += 1; }

  async run(inputs: I, fetchResult: (inputs: I) => Promise<R>, publish: (state: ShotRequestState<I, R>) => void) {
    const version = ++this.version;
    const snapshot = { ...inputs };
    const state = { ...emptyShotResult<I, R>(), key: shotInputKey(snapshot), inputs: snapshot };
    publish({ ...state, loading: true });
    try {
      const result = await fetchResult(snapshot);
      if (version === this.version) publish({ ...state, result });
    } catch (reason) {
      if (version === this.version) publish({ ...state, error: reason instanceof Error ? reason.message : "The request could not be completed." });
    }
  }
}
