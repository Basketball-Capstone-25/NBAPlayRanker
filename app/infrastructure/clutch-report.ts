// Infrastructure Layer — clutch report API (GET /data/clutch-report), SCRUM-629.
import { createClient } from "../../lib/supabase/client";
import type { ClutchFilters, ClutchReport } from "../domain/clutch-report";
import { API_BASE } from "./api-client";

export type ClutchFormat = "pdf" | "csv";

/** The signed-in user is not an analyst (HTTP 401/403). */
export class ClutchAccessError extends Error {}

function clutchParams(f: ClutchFilters): URLSearchParams {
  return new URLSearchParams({
    season: f.season.trim(),
    team: f.team.trim().toUpperCase(),
    opp: f.opp.trim().toUpperCase(),
    period_from: String(f.periodFrom),
    seconds_left: String(f.secondsLeft),
    margin_min: String(f.marginMin),
    margin_max: String(f.marginMax),
  });
}

export function clutchReportUrl(f: ClutchFilters, format?: ClutchFormat): string {
  return `${API_BASE}/data/clutch-report${format ? `.${format}` : ""}?${clutchParams(f)}`;
}

export function clutchFilename(f: ClutchFilters, format: ClutchFormat): string {
  return `clutch_${f.team.toUpperCase()}_vs_${f.opp.toUpperCase()}_${f.season}.${format}`;
}

export async function fetchClutchReport(f: ClutchFilters, signal?: AbortSignal): Promise<ClutchReport> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(clutchReportUrl(f), { headers, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (response.status === 401 || response.status === 403) throw new ClutchAccessError("Only analysts can build clutch reports.");
    throw new Error(detail ?? `The clutch report could not be built (${response.status}).`);
  }
  return response.json();
}
