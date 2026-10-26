// Infrastructure Layer — analyst request log (GET /admin/request-log), SCRUM-494.
import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";

export type RequestLogFilters = {
  user?: string;
  endpoint?: string;
  dateFrom?: string; // YYYY-MM-DD
  dateTo?: string; // YYYY-MM-DD
  page?: number;
  pageSize?: number;
};

export type RequestLogEntry = {
  id: number;
  created_at: string;
  user_email: string | null;
  user_role: "coach" | "analyst";
  method: string;
  endpoint: string;
  query: string;
  status_code: number;
  duration_ms: number;
};

export type RequestLogPage = {
  items: RequestLogEntry[];
  page: number;
  page_size: number;
  total: number | null;
  has_next: boolean;
};

export const REQUEST_LOG_PAGE_SIZE = 50;

/** The signed-in user is not an analyst (HTTP 401/403). */
export class RequestLogForbiddenError extends Error {}

export function requestLogUrl(filters: RequestLogFilters): string {
  const query = new URLSearchParams();
  if (filters.user?.trim()) query.set("user", filters.user.trim());
  if (filters.endpoint?.trim()) query.set("endpoint", filters.endpoint.trim());
  if (filters.dateFrom) query.set("date_from", filters.dateFrom);
  if (filters.dateTo) query.set("date_to", filters.dateTo);
  query.set("page", String(filters.page ?? 1));
  query.set("page_size", String(filters.pageSize ?? REQUEST_LOG_PAGE_SIZE));
  return `${API_BASE}/admin/request-log?${query}`;
}

export async function fetchRequestLog(filters: RequestLogFilters, signal?: AbortSignal): Promise<RequestLogPage> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(requestLogUrl(filters), { headers, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (response.status === 401 || response.status === 403) {
      throw new RequestLogForbiddenError(detail ?? "Only analysts can view the request log.");
    }
    throw new Error(detail ?? `Request log failed (${response.status}).`);
  }
  return response.json();
}
