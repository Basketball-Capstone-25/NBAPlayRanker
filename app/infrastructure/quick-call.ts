// Infrastructure Layer — Quick-Call API client (GET /rank-plays/quick-call).
import { createClient } from "../../lib/supabase/client";
import type { QuickCallSituation } from "../domain/quick-call";
import { API_BASE } from "./api-client";

export type QuickCallPlay = {
  rank: number;
  play_type: string;
  ppp: number;
  shot_zone: string;
  reason: string;
};

export type QuickCallResponse = {
  source: "context" | "baseline";
  notice: string;
  plays: QuickCallPlay[];
  situation: {
    season: string;
    our_team: string;
    opp_team: string;
    margin: number;
    period: number;
    time_remaining: number;
  };
  k: number;
  elapsed_ms: number;
};

/** The API refused the request because the signed-in user is not a coach. */
export class QuickCallForbiddenError extends Error {}

export function quickCallUrl(s: QuickCallSituation): string {
  const query = new URLSearchParams({
    season: s.season,
    our: s.ourTeam,
    opp: s.oppTeam,
    margin: String(s.margin),
    period: String(s.period),
    time_remaining: String(s.timeRemaining),
  });
  return `${API_BASE}/rank-plays/quick-call?${query}`;
}

export async function fetchQuickCall(s: QuickCallSituation, signal?: AbortSignal): Promise<QuickCallResponse> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(quickCallUrl(s), { headers, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (response.status === 401 || response.status === 403) {
      throw new QuickCallForbiddenError(detail ?? "Quick-Call Mode is for signed-in coaches.");
    }
    throw new Error(detail ?? `Quick-call request failed (${response.status}).`);
  }
  return response.json();
}
