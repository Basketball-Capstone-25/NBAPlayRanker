// Infrastructure Layer — Data Explorer play-by-play shots (GET /data/pbp-shots), SCRUM-627.
import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";

export type ShotResult = "" | "made" | "missed";

export type ShotFilters = {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  result?: ShotResult;
};

export type ShotColumn = { key: string; label: string };
export type ShotRow = Record<string, string | number | null>;

export type ShotPage = {
  columns: ShotColumn[];
  rows: ShotRow[];
  page: number;
  page_size: number;
  total: number;
  has_next: boolean;
  dataset_version: string | null;
};

export const SHOT_PAGE_SIZE = 50;

/** The signed-in user is not an analyst (HTTP 401/403). */
export class ShotAccessError extends Error {}

/** The shot dataset has not been built or cannot be read (HTTP 503). */
export class ShotDataUnavailableError extends Error {}

function filterParams(filters: ShotFilters): URLSearchParams {
  const query = new URLSearchParams();
  query.set("season", filters.season.trim());
  query.set("team", filters.team.trim());
  if (filters.opp?.trim()) query.set("opp", filters.opp.trim());
  if (filters.shotType?.trim()) query.set("shot_type", filters.shotType.trim());
  if (filters.zone?.trim()) query.set("zone", filters.zone.trim());
  if (filters.result) query.set("result", filters.result);
  return query;
}

/** True when a search can be sent: the API needs both a season and a team. */
export function canSearchShots(filters: ShotFilters): boolean {
  return Boolean(filters.season.trim() && filters.team.trim());
}

export function shotExplorerUrl(filters: ShotFilters, page: number): string {
  const query = filterParams(filters);
  query.set("page", String(page));
  query.set("page_size", String(SHOT_PAGE_SIZE));
  return `${API_BASE}/data/pbp-shots?${query}`;
}

export function shotExportUrl(filters: ShotFilters, format: "csv" | "json"): string {
  return `${API_BASE}/data/pbp-shots.${format}?${filterParams(filters)}`;
}

export function shotExportFilename(filters: ShotFilters, format: "csv" | "json"): string {
  return `pbp_shots_${filters.season}_${filters.team}_${filters.opp || "all"}.${format}`;
}

export async function fetchShotPage(filters: ShotFilters, page: number, signal?: AbortSignal): Promise<ShotPage> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(shotExplorerUrl(filters, page), { headers, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (response.status === 401 || response.status === 403) {
      throw new ShotAccessError("Only analysts can view this data.");
    }
    if (response.status === 503) throw new ShotDataUnavailableError(detail ?? "Shot data is not available yet.");
    throw new Error(detail ?? `Shot search failed (${response.status}).`);
  }
  return response.json();
}
