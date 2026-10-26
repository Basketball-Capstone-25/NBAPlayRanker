import { createClient } from "../../../../lib/supabase/server";
import { parseShotHeatmapExport, shotHeatmapFilename } from "../../../domain/shot-heatmap-export";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

const PRIVATE_HEADERS = { "Cache-Control": "private, no-store", Pragma: "no-cache", "X-Content-Type-Options": "nosniff" };
const MAX_JSON_BYTES = 4 * 1024 * 1024;
const MAX_PNG_BYTES = 3 * 1024 * 1024;
const COUNT_KEYS = ["n_shots_total", "n_shots_rendered", "n_shots_excluded_location", "n_shots_outside_halfcourt", "n_shots_omitted_by_cap"] as const;

function error(message: string, status: number) {
  return Response.json({ error: message }, { status, headers: PRIVATE_HEADERS });
}

async function readHeatmap(response: Response): Promise<Record<string, unknown>> {
  if (!response.body || Number(response.headers.get("content-length")) > MAX_JSON_BYTES) throw new Error("Invalid response size");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_JSON_BYTES) { await reader.cancel(); throw new Error("Response too large"); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const raw: unknown = JSON.parse(Buffer.concat(chunks, size).toString("utf8"));
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Invalid response");
  return raw as Record<string, unknown>;
}

function decodePng(value: unknown): Buffer {
  if (typeof value !== "string" || !value || value.length > Math.ceil(MAX_PNG_BYTES / 3) * 4 ||
      value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw new Error("Invalid image encoding");
  const bytes = Buffer.from(value, "base64");
  if (bytes.length > MAX_PNG_BYTES || bytes.toString("base64") !== value || bytes.length < 45 ||
      !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error("Invalid PNG signature");
  let offset = 8;
  let hasImageData = false;
  while (offset + 12 <= bytes.length) {
    const size = bytes.readUInt32BE(offset);
    const type = bytes.toString("ascii", offset + 4, offset + 8);
    const next = offset + size + 12;
    if (next > bytes.length) throw new Error("Truncated PNG");
    if (offset === 8) {
      if (type !== "IHDR" || size !== 13) throw new Error("Missing PNG header");
      const width = bytes.readUInt32BE(offset + 8), height = bytes.readUInt32BE(offset + 12);
      if (!width || !height || width > 4096 || height > 4096) throw new Error("Invalid image dimensions");
    } else if (type === "IHDR") throw new Error("Duplicate PNG header");
    if (type === "IDAT" && size > 0) hasImageData = true;
    if (type === "IEND") {
      if (size !== 0 || next !== bytes.length || !hasImageData) throw new Error("Invalid PNG end");
      return bytes;
    }
    offset = next;
  }
  throw new Error("Missing PNG end");
}

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const origin = request.headers.get("origin");
  if ((origin && origin !== incoming.origin) || request.headers.get("sec-fetch-site") === "cross-site") {
    return error("Cross-origin export requests are not allowed.", 403);
  }
  const inputs = parseShotHeatmapExport(incoming.searchParams);
  if (!inputs) return error("Render the heatmap again before downloading.", 400);

  try {
    const supabase = await createClient();
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) return error("Sign in to download this image.", 401);
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    if (sessionError || !session?.access_token || session.user.id !== user.id) return error("Sign in to download this image.", 401);
    const { data: profile, error: profileError } = await supabase.from("profiles").select("role").eq("id", user.id).maybeSingle();
    if (profileError || !profile || !["analyst", "coach"].includes(profile.role)) return error("This account cannot download shot heatmaps.", 403);

    const target = new URL("/pbp/viz/shot-heatmap", process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000");
    target.search = new URLSearchParams({ season: inputs.season, our: inputs.team, opp: inputs.opp, max_shots: String(inputs.maxShots) }).toString();
    if (inputs.shotType) target.searchParams.set("shot_type", inputs.shotType);
    if (inputs.zone) target.searchParams.set("zone", inputs.zone);
    const upstream = await fetch(target, {
      headers: { Authorization: `Bearer ${session.access_token}` }, cache: "no-store", redirect: "manual",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(55000)]),
    });
    if (!upstream.ok) {
      const status = (upstream.status >= 400 && upstream.status < 500) || upstream.status === 503 ? upstream.status : 502;
      return error(upstream.status === 404 ? "No shot locations match these filters. Change the filters and render again." : "The heatmap could not be downloaded. Check your access and try again.", status);
    }
    if (upstream.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
      return error("The heatmap service returned an unexpected response.", 502);
    }
    const data = await readHeatmap(upstream);
    if (typeof data.dataset_version !== "string" || !/^sha256:[a-f0-9]{64}$/.test(data.dataset_version)) return error("The image source could not be verified.", 502);
    if (data.dataset_version !== inputs.version) return error("The shot data changed. Render the heatmap again before downloading.", 409);
    if (data.season !== inputs.season || data.team !== inputs.team || data.opp !== inputs.opp ||
        (data.shot_type ?? "") !== inputs.shotType || (data.zone ?? "") !== inputs.zone || data.max_shots !== inputs.maxShots ||
        COUNT_KEYS.some(key => !Number.isSafeInteger(data[key]) || Number(data[key]) < 0 || Number(data[key]) > 10000000)) {
      return error("The image does not match the selected filters.", 502);
    }
    const total = Number(data.n_shots_total), shown = Number(data.n_shots_rendered);
    if (shown > inputs.maxShots || data.n_shots !== shown || (Number(data.n_shots_omitted_by_cap) > 0 && shown !== inputs.maxShots) || total !== shown + Number(data.n_shots_excluded_location) +
        Number(data.n_shots_outside_halfcourt) + Number(data.n_shots_omitted_by_cap)) return error("The image counts could not be verified.", 502);
    const bytes = decodePng(data.image_base64);
    return new Response(new Uint8Array(bytes), { headers: {
      ...PRIVATE_HEADERS, "Content-Type": "image/png", "Content-Length": String(bytes.length),
      "Content-Disposition": `attachment; filename="${shotHeatmapFilename(inputs)}"`,
      "X-Dataset-Version": data.dataset_version, "X-Shots-Matched": String(total), "X-Shots-Shown": String(shown),
    } });
  } catch {
    return error("Unable to download the heatmap. Please try again.", 502);
  }
}
