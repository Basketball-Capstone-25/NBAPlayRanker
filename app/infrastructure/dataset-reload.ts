// Infrastructure layer — asks the server to reload the stored play-type or shot files.

import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";
import { DatasetReloadError, parseDatasetReload, type DatasetReloadResult, type ReloadDataset } from "../domain/dataset-reload";

const paths: Record<ReloadDataset, string> = { synergy: "/data/reload/synergy", pbp: "/data/reload/pbp" };
const unknownResult = () => new DatasetReloadError(
  "Could not confirm the reload result. It may still be running. Check the data before trying again.", "unknown",
);

/** Reload an existing server file once; a lost reply must never trigger another POST. */
export async function reloadDataset(dataset: ReloadDataset, signal?: AbortSignal): Promise<DatasetReloadResult> {
  const path = paths[dataset];
  if (dataset !== "synergy" && dataset !== "pbp") throw new DatasetReloadError("Choose a supported dataset.", "not_started");
  signal?.throwIfAborted();
  let token: string | undefined;
  try {
    const { data, error } = await createClient().auth.getSession();
    if (!error) token = data.session?.access_token;
  } catch {
    // A missing session or client configuration must not send a reload request.
  }
  signal?.throwIfAborted();
  if (!token) throw new DatasetReloadError("Sign in again to reload data.", "not_started");
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST", cache: "no-store", signal, headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    throw unknownResult();
  }
  if (response.status === 401) throw new DatasetReloadError("Sign in again to reload data.", "not_started", 401);
  if (response.status === 403) throw new DatasetReloadError("An analyst account is required to reload data.", "not_started", 403);
  if (response.status === 404) throw new DatasetReloadError("This dataset reload is not available on the server yet.", "not_started", 404);
  if (response.status === 409) throw new DatasetReloadError("A reload is already running. Wait for it to finish before trying again.", "rejected", 409);
  if (response.status === 422) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    // The reload service supplies a short, safe reason. Do not expose arbitrary
    // validation objects, HTML errors or server traces as a user-facing message.
    const reason = typeof detail === "string" && detail.length <= 500
      && detail.startsWith("Reload failed; the previous data is still in use. ")
      && !/[<>\u0000-\u001f\u007f]/.test(detail) ? detail
      : "Reload failed. The previous data is still in use.";
    throw new DatasetReloadError(reason, "rejected", 422);
  }
  if (!response.ok) throw new DatasetReloadError(
    "The reload service is unavailable. Check the data before trying again.", "unknown", response.status,
  );
  let body: unknown;
  try { body = await response.json(); } catch { throw unknownResult(); }
  return parseDatasetReload(body, dataset);
}
