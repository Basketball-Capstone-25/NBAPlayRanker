import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";

export type UpliftFilters = { season: string; our: string; opp: string; k: number; wOff?: number };

export type UpliftRow = {
  rank: number;
  play_type: string;
  modeled_ppp: number;
  historical_offense_possessions: number;
  topk_weight: number;
  uplift_vs_team_season_ppp: number;
};

export type TopKUpliftReport = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  k_returned: number;
  evaluation_scope: string;
  team_season: { ppp: number; possessions_denominator: number };
  topk: { modeled_ppp: number; possessions_denominator: number };
  uplift_ppp: number;
  uplift_percent: number | null;
  rankings: UpliftRow[];
};

/** Thrown when the API has no uplift result for the chosen matchup (HTTP 404). */
export class UpliftUnavailableError extends Error {}

export function topKUpliftUrl(filters: UpliftFilters): string {
  const query = new URLSearchParams({
    season: filters.season, our: filters.our, opp: filters.opp,
    k: String(filters.k), w_off: String(filters.wOff ?? 0.7),
  });
  return `${API_BASE}/metrics/topk-uplift?${query}`;
}

export async function fetchTopKUplift(filters: UpliftFilters, signal?: AbortSignal): Promise<TopKUpliftReport> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(topKUpliftUrl(filters), { headers, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (response.status === 404) throw new UpliftUnavailableError(detail ?? "No uplift result for this matchup.");
    throw new Error(detail ?? `Uplift request failed (${response.status}).`);
  }
  return response.json();
}
