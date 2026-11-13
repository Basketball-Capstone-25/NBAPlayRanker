import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { parseLeagueComparison, type LeagueComparison, type LeagueFilters } from "../domain/league-comparison";

export async function fetchLeagueComparison(filters: LeagueFilters, signal?: AbortSignal): Promise<LeagueComparison> {
  const snapshot = { ...filters };
  signal?.throwIfAborted();
  const session = await createClient().auth.getSession().catch(() => null);
  signal?.throwIfAborted();
  const token = session?.data.session?.access_token;
  if (session?.error || !token) throw new Error("Sign in again to view the league comparison.");
  const query = new URLSearchParams(snapshot);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes/league-comparison?${query}`, {
      cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the comparison service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to view the league comparison.");
  if (response.status === 403) throw new Error("A coach account is required to view this comparison.");
  if (response.status === 400 || response.status === 422) throw new Error("Check the team, season and side, then try again.");
  if (!response.ok) throw new Error("The comparison service is unavailable. Please try again.");
  const body = await response.json().catch(() => {
    signal?.throwIfAborted();
    throw new Error("The league comparison is incomplete. Please try again.");
  });
  signal?.throwIfAborted();
  return parseLeagueComparison(body, snapshot);
}
