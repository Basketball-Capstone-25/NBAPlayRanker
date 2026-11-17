import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { parseTeamComparison, type TeamComparison, type TeamComparisonFilters } from "../domain/team-comparison";

export async function fetchTeamComparison(filters: TeamComparisonFilters, signal?: AbortSignal): Promise<TeamComparison> {
  const snapshot = { season: filters.season, team: filters.team, opponent: filters.opponent };
  signal?.throwIfAborted();
  const session = await createClient().auth.getSession().catch(() => null);
  signal?.throwIfAborted();
  const token = session?.data.session?.access_token;
  if (session?.error || !token) throw new Error("Sign in again to compare team profiles.");
  const query = new URLSearchParams(snapshot);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes/team-comparison?${query}`, {
      cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the comparison service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to compare team profiles.");
  if (response.status === 403) throw new Error("A coach account is required to compare team profiles.");
  if (response.status === 400 || response.status === 422) throw new Error("Choose two different teams and an available season.");
  if (!response.ok) throw new Error("The comparison service is unavailable. Please try again.");
  const body = await response.json().catch(() => {
    signal?.throwIfAborted();
    throw new Error("The team comparison is incomplete. Please try again.");
  });
  signal?.throwIfAborted();
  return parseTeamComparison(body, snapshot);
}
