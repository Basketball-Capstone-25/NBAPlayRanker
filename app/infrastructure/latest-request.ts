/** Deduplicate an in-flight resource request and ignore superseded results. */
export class LatestRequest {
  private version = 0;
  private pending: { key: string; promise: Promise<void> } | null = null;

  run<T>(key: string, request: () => Promise<T>, handlers: {
    start: () => void; success: (value: T) => void; error: (error: unknown) => void; settled: () => void;
  }, force = false): Promise<void> {
    if (!force && this.pending?.key === key) return this.pending.promise;
    const version = ++this.version;
    handlers.start();
    const promise = (async () => {
      try {
        // Let the pending slot be assigned even when a request throws immediately.
        await Promise.resolve();
        const value = await request();
        if (version === this.version) handlers.success(value);
      } catch (error) {
        if (version === this.version) handlers.error(error);
      } finally {
        if (version === this.version) {
          this.pending = null;
          handlers.settled();
        }
      }
    })();
    this.pending = { key, promise };
    return promise;
  }
}
