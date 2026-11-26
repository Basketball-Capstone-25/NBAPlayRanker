// Fetches usage and defense rows and tracks which dashboard request owns them.

import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { dashboardKey, parsePlaytypeDashboard, type DashboardFilters, type PlaytypeDashboard } from "../domain/playtype-dashboard";

/** Reject after one minute so the caller can offer a retry. The caller owns cancellation of the underlying request. */
export async function withRequestDeadline<T>(pending: Promise<T>, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([pending, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), 60000);
    })]);
  } finally { clearTimeout(timer); }
}

/** Send the selected matchup and cutoff, then validate that the response belongs to those filters. */
export async function fetchPlaytypeDashboard(filters: DashboardFilters, signal?: AbortSignal): Promise<PlaytypeDashboard> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const query = new URLSearchParams({ season: filters.season, our: filters.our, opp: filters.opp, min_poss: String(filters.min_poss) });
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes?${query}`, {
      cache: "no-store", signal, headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the comparison service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to view the matchup dashboard.");
  if (response.status === 403) throw new Error("A coach account is required to view this dashboard.");
  if (response.status === 404) throw new Error("No regular-season data is available for this selection.");
  if (response.status === 400 || response.status === 422) throw new Error("Check the teams, season and minimum possessions, then try again.");
  if (!response.ok) throw new Error("The comparison service is unavailable. Please try again.");
  const body = await response.json().catch(() => { throw new Error("The comparison data is incomplete. Please try again."); });
  return parsePlaytypeDashboard(body, filters);
}

export type DashboardState = { key: string; data: PlaytypeDashboard | null; loading: boolean; error: string | null };
/** Start with a loading state until the submitted selection has a matching reply. */
export const emptyDashboard = (): DashboardState => ({ key: "", data: null, loading: true, error: null });
/** Hide rows from an older selection while a different matchup is loading. */
export const selectDashboard = (state: DashboardState, filters: DashboardFilters): DashboardState =>
  state.key === dashboardKey(filters) ? state : emptyDashboard();

/** Publish results only for the latest submission, including when an earlier request finishes late. */
export class DashboardRequest {
  private controller: AbortController | null = null;
  private version = 0;
  /** Invalidate outstanding updates and abort the active network request. */
  cancel() { this.version++; this.controller?.abort(); }

  /** Snapshot the inputs and publish loading, success or timeout for that selection. */
  async run(filters: DashboardFilters,
    fetchResult: (filters: DashboardFilters, signal: AbortSignal) => Promise<PlaytypeDashboard>,
    publish: (state: DashboardState) => void) {
    this.cancel();
    const version = this.version, snapshot = { ...filters }, key = dashboardKey(snapshot);
    const controller = new AbortController();
    this.controller = controller;
    publish({ key, data: null, loading: true, error: null });
    const timer = setTimeout(() => {
      if (version === this.version) {
        controller.abort();
        publish({ key, data: null, loading: false, error: "The comparison took too long. Please try again." });
      }
    }, 60000);
    try {
      const data = await fetchResult(snapshot, controller.signal);
      if (version === this.version && !controller.signal.aborted) publish({ key, data, loading: false, error: null });
    } catch (error) {
      if (version === this.version && !controller.signal.aborted) publish({ key, data: null, loading: false,
        error: error instanceof Error ? error.message : "Could not load the comparison. Please try again." });
    } finally { clearTimeout(timer); }
  }
}
