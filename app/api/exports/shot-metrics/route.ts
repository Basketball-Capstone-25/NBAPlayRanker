import { createClient } from "../../../../lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

const PRIVATE_HEADERS = { "Cache-Control": "private, no-store", Pragma: "no-cache", "X-Content-Type-Options": "nosniff" };
const FIELDS = ["RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"] as const;
const MAX_BODY = 256 * 1024;

function error(message: string, status: number) {
  return Response.json({ error: message }, { status, headers: PRIVATE_HEADERS });
}

async function readSavedResult(response: Response): Promise<unknown> {
  if (!response.body) throw new Error("Empty result");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) { await reader.cancel(); throw new Error("Result too large"); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return JSON.parse(new TextDecoder().decode(bytes));
}

function csvText(value: string): string {
  const safe = /^[\s]*[=+\-@\t\r\n]/.test(value) ? `'${value}` : value;
  return `"${safe.replaceAll('"', '""')}"`;
}

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const origin = request.headers.get("origin");
  if ((origin && origin !== incoming.origin) || request.headers.get("sec-fetch-site") === "cross-site") {
    return error("Cross-origin export requests are not allowed.", 403);
  }
  const expectedVersion = incoming.searchParams.get("version");
  if (!expectedVersion || !/^[a-f0-9]{64}$/.test(expectedVersion) ||
      [...incoming.searchParams.keys()].some(key => key !== "version") || incoming.searchParams.getAll("version").length !== 1) {
    return error("Reload the saved results before exporting.", 400);
  }

  try {
    const supabase = await createClient();
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) return error("Sign in to download this file.", 401);
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    if (sessionError || !session?.access_token || session.user.id !== user.id) return error("Sign in to download this file.", 401);

    const upstream = await fetch(new URL("/metrics/shot-models?n_splits=5", process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"), {
      headers: { Authorization: `Bearer ${session.access_token}` }, cache: "no-store", redirect: "manual", signal: request.signal,
    });
    if (!upstream.ok) {
      const status = (upstream.status >= 400 && upstream.status < 500) || upstream.status === 503 ? upstream.status : 502;
      return error("Saved results could not be downloaded. Check your access and reload the page.", status);
    }
    if (upstream.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
      return error("The saved-results service returned an unexpected response.", 502);
    }
    const raw = await readSavedResult(upstream);
    if (!raw || typeof raw !== "object") return error("The saved results are incomplete.", 502);
    const data = raw as Record<string, unknown>;
    if (data.n_splits !== 5 || data.cached !== true || typeof data.source_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(data.source_sha256) ||
        !Array.isArray(data.metrics) || data.metrics.length < 1 || data.metrics.length > 10) {
      return error("The saved results are incomplete.", 502);
    }
    if (data.source_sha256 !== expectedVersion) return error("The shot data changed. Reload the results before exporting.", 409);
    const version = `sha256:${data.source_sha256}`;
    const lines = [["model", ...FIELDS, "n_splits", "DATASET_VERSION"].join(",")];
    const models = new Set<string>();
    for (const value of data.metrics) {
      if (!value || typeof value !== "object") return error("The saved results are incomplete.", 502);
      const row = value as Record<string, unknown>;
      if (typeof row.model !== "string" || !row.model.trim() || row.model.length > 80 || models.has(row.model) ||
          FIELDS.some(field => typeof row[field] !== "number" || !Number.isFinite(row[field]))) {
        return error("The saved results contain invalid metrics.", 502);
      }
      models.add(row.model);
      lines.push([csvText(row.model), ...FIELDS.map(field => String(row[field])), "5", csvText(version)].join(","));
    }
    return new Response(lines.join("\r\n") + "\r\n", { headers: {
      ...PRIVATE_HEADERS, "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="shot_model_metrics_splits_5.csv"', "X-Dataset-Version": version,
    } });
  } catch {
    return error("Unable to download the saved results. Please try again.", 502);
  }
}
