// Retrieves outcome rates and source coverage for the selected offense.

import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { parsePossessionOutcomes, type OutcomeFilters, type PossessionOutcomes } from "../domain/possession-outcomes";

/** Validate the returned team, season and rate coverage before handing the rows to the chart. */
export async function fetchPossessionOutcomes(filters: OutcomeFilters, signal?: AbortSignal): Promise<PossessionOutcomes> {
  const snapshot = { season: filters.season, team: filters.team };
  signal?.throwIfAborted();
  const session = await createClient().auth.getSession().catch(() => null);
  signal?.throwIfAborted();
  const token = session?.data.session?.access_token;
  if (session?.error || !token) throw new Error("Sign in again to view possession rates.");
  const query = new URLSearchParams(snapshot);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes/possession-outcomes?${query}`, {
      cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the possession-rate service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to view possession rates.");
  if (response.status === 403) throw new Error("A coach account is required to view possession rates.");
  if (response.status === 400 || response.status === 422) throw new Error("Choose a team and an available season.");
  if (!response.ok) throw new Error("The possession-rate service is unavailable. Please try again.");
  const body = await response.json().catch(() => {
    signal?.throwIfAborted();
    throw new Error("The possession rates are incomplete. Please try again.");
  });
  signal?.throwIfAborted();
  return parsePossessionOutcomes(body, snapshot);
}
