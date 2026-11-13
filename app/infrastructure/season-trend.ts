import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { parseSeasonTrend, type SeasonTrend, type TrendFilters } from "../domain/season-trend";

export async function fetchSeasonTrend(filters: TrendFilters, signal?: AbortSignal): Promise<SeasonTrend> {
  const snapshot = { ...filters };
  signal?.throwIfAborted();
  const session = await createClient().auth.getSession().catch(() => null);
  signal?.throwIfAborted();
  const token = session?.data.session?.access_token;
  if (session?.error || !token) throw new Error("Sign in again to view the season trend.");
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analytics/playtypes/season-trend?${new URLSearchParams(snapshot)}`, {
      cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Could not reach the season trend service. Please try again.");
  }
  if (response.status === 401) throw new Error("Sign in again to view the season trend.");
  if (response.status === 403) throw new Error("A coach account is required to view this trend.");
  if (response.status === 400 || response.status === 422) throw new Error("Check the team and play type, then try again.");
  if (!response.ok) throw new Error("The season trend service is unavailable. Please try again.");
  const body = await response.json().catch(() => {
    signal?.throwIfAborted();
    throw new Error("The season trend is incomplete. Please try again.");
  });
  signal?.throwIfAborted();
  return parseSeasonTrend(body, snapshot);
}
