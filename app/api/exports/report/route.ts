import { createClient } from "../../../../lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_PAYLOAD = 256 * 1024;
const MAX_FILE = 2 * 1024 * 1024;
const MEDIA = { csv: "text/csv", json: "application/json", pdf: "application/pdf" } as const;
const REPORTS = new Set(["baseline_rankings", "context_rankings", "model_metrics"]);
const PRIVATE_HEADERS = { "Cache-Control": "private, no-store", Pragma: "no-cache", "X-Content-Type-Options": "nosniff" };

function error(message: string, status: number) {
  return Response.json({ error: message }, { status, headers: PRIVATE_HEADERS });
}

async function readLimited(stream: ReadableStream<Uint8Array> | null, limit: number): Promise<Uint8Array> {
  if (!stream) throw new Error("empty body");
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) { await reader.cancel(); throw new Error("body too large"); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const output = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

export async function POST(request: Request) {
  const incoming = new URL(request.url);
  if (request.headers.get("origin") !== incoming.origin || request.headers.get("sec-fetch-site") === "cross-site") {
    return error("Cross-origin export requests are not allowed.", 403);
  }
  if (incoming.search) return error("Invalid export parameters.", 400);
  const contentType = request.headers.get("content-type")?.split(";")[0].trim().toLowerCase();
  const preflight = contentType === "application/json";
  if (!preflight && contentType !== "application/x-www-form-urlencoded") return error("Unsupported request format.", 415);

  let body: { format: keyof typeof MEDIA; payload: { report: string; rows: unknown[] } };
  let encoded: string;
  try {
    const bytes = await readLimited(request.body, preflight ? MAX_PAYLOAD : MAX_PAYLOAD * 3 + 100);
    const text = new TextDecoder().decode(bytes);
    if (preflight) encoded = text;
    else {
      const form = new URLSearchParams(text);
      if ([...form.keys()].some(key => key !== "request") || form.getAll("request").length !== 1) return error("Invalid export parameters.", 400);
      encoded = form.get("request")!;
    }
    if (new TextEncoder().encode(encoded).byteLength > MAX_PAYLOAD) return error("This report is too large to export.", 413);
    body = JSON.parse(encoded);
    if (!body || !Object.hasOwn(MEDIA, body.format) || !REPORTS.has(body.payload?.report) || !Array.isArray(body.payload?.rows) || !body.payload.rows.length) {
      return error("Run a supported report before exporting.", 400);
    }
  } catch (caught) {
    return error(caught instanceof Error && caught.message === "body too large" ? "This report is too large to export." : "Invalid export request.", caught instanceof Error && caught.message === "body too large" ? 413 : 400);
  }

  try {
    const supabase = await createClient();
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) return error("Sign in again to export this report.", 401);
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    if (sessionError || !session?.access_token || session.user.id !== user.id) return error("Sign in again to export this report.", 401);
    const upstream = await fetch(new URL("/export/report", process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"), {
      method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.access_token}` },
      body: encoded, cache: "no-store", redirect: "manual", signal: request.signal,
    });
    if (!upstream.ok) {
      if (upstream.status === 401) return error("Sign in again to export this report.", 401);
      if (upstream.status === 403) return error("Your account does not have access to this report.", 403);
      if (upstream.status >= 400 && upstream.status < 500) return error("The report could not be exported. Run it again and retry.", upstream.status);
      return error("The export service is unavailable. Please try again.", 502);
    }
    const responseType = upstream.headers.get("content-type") || "";
    const serverName = upstream.headers.get("content-disposition")?.match(/filename="([^"]+)"/)?.[1];
    if (responseType.split(";")[0].trim().toLowerCase() !== MEDIA[body.format] || !serverName ||
      serverName.length > 200 || !serverName.endsWith(`.${body.format}`) || /[\r\n/\\]/.test(serverName)) {
      return error("The export service returned an unexpected file. Please try again.", 502);
    }
    const file = await readLimited(upstream.body, MAX_FILE);
    if (!file.length) return error("The export service returned an empty file.", 502);
    if (body.format === "pdf" && new TextDecoder().decode(file.slice(0, 5)) !== "%PDF-") return error("The export service returned an invalid PDF.", 502);
    if (body.format === "json") {
      const payload = JSON.parse(new TextDecoder().decode(file));
      if (payload.contract_version !== "1" || payload.report !== body.payload.report || !Array.isArray(payload.rows) || !payload.rows.length) {
        return error("The export service returned an invalid report.", 502);
      }
    }
    if (preflight) return Response.json({ ready: true }, { headers: PRIVATE_HEADERS });
    const filename = serverName.replace(/[^a-zA-Z0-9._ -]/g, "_");
    return new Response(new Uint8Array(file).buffer, { headers: { ...PRIVATE_HEADERS, "Content-Type": responseType, "Content-Disposition": `attachment; filename="${filename}"` } });
  } catch {
    return error("Could not prepare this export. Please try again.", 502);
  }
}
