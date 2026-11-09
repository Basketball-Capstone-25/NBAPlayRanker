import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { parseSeasonSummary, summaryKey, type SeasonSummary, type SummaryFilters } from "../domain/season-summary";

export async function fetchSeasonSummary(filters: SummaryFilters, signal?: AbortSignal): Promise<SeasonSummary> {
  const session = await createClient().auth.getSession().catch(() => null);
  signal?.throwIfAborted();
  const token = session?.data.session?.access_token;
  if (session?.error || !token) throw new Error("Sign in again to view the season summary.");
  const query = new URLSearchParams({ period: "season", season: filters.season, our: filters.our,
    opp: filters.opp, k: String(filters.k), w_off: String(filters.w_off) });
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes/summary?${query}`, {
      cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the summary service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to view the season summary.");
  if (response.status === 403) throw new Error("A coach or analyst account is required to view this summary.");
  if (response.status === 400 || response.status === 422) throw new Error("Check the teams, season and Top-K settings, then try again.");
  if (!response.ok) throw new Error("The summary service is unavailable. Please try again.");
  const body = await response.json().catch(() => { throw new Error("The season summary is incomplete. Please try again."); });
  return parseSeasonSummary(body, filters);
}

export type SummaryState = { key: string; data: SeasonSummary | null; loading: boolean; error: string | null };
export const emptySummary = (): SummaryState => ({ key: "", data: null, loading: true, error: null });
export const selectSummary = (state: SummaryState, filters: SummaryFilters): SummaryState =>
  state.key === summaryKey(filters) ? state : emptySummary();

export class SummaryRequest {
  private controller: AbortController | null = null;
  private version = 0;
  cancel() { this.version++; this.controller?.abort(); }

  async run(filters: SummaryFilters,
    fetchResult: (filters: SummaryFilters, signal: AbortSignal) => Promise<SeasonSummary>,
    publish: (state: SummaryState) => void) {
    this.cancel();
    const version = this.version, snapshot = { ...filters }, key = summaryKey(snapshot);
    const controller = new AbortController();
    this.controller = controller;
    publish({ key, data: null, loading: true, error: null });
    let timedOut = false, onAbort: () => void;
    const cancelled = new Promise<never>((_, reject) => {
      onAbort = () => reject(new Error("Request cancelled"));
      controller.signal.addEventListener("abort", onAbort, { once: true });
    });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 60000);
    try {
      const data = await Promise.race([fetchResult(snapshot, controller.signal), cancelled]);
      if (version === this.version && !controller.signal.aborted) publish({ key, data, loading: false, error: null });
    } catch (error) {
      if (version === this.version && (timedOut || !controller.signal.aborted)) publish({
        key, data: null, loading: false,
        error: timedOut ? "The season summary took too long. Please try again."
          : error instanceof Error ? error.message : "Could not load the summary. Please try again.",
      });
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener("abort", onAbort);
      if (this.controller === controller) this.controller = null;
    }
  }
}
