import { createClient } from "../../../lib/supabase/server";
import { EXPORT_PATHS, exportFilename } from "../../infrastructure/export-contract";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

const PRIVATE_HEADERS = {
  "Cache-Control": "private, no-store",
  "Pragma": "no-cache",
  "X-Content-Type-Options": "nosniff",
};

function errorResponse(message: string, status: number) {
  // Error responses deliberately have no attachment header.
  return Response.json({ error: message }, { status, headers: PRIVATE_HEADERS });
}

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const origin = request.headers.get("origin");
  if ((origin && origin !== incoming.origin) || request.headers.get("sec-fetch-site") === "cross-site") {
    return errorResponse("Cross-origin export requests are not allowed.", 403);
  }
  const path = incoming.searchParams.get("path") || "";
  if (!EXPORT_PATHS.has(path)) return errorResponse("Unsupported export path.", 400);
  for (const name of incoming.searchParams.keys()) {
    if (!["path", "query", "filename"].includes(name) || incoming.searchParams.getAll(name).length !== 1) {
      return errorResponse("Invalid export parameters.", 400);
    }
  }

  try {
    const supabase = await createClient();
    // Validate the cookie's identity with Auth before using its access token.
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) return errorResponse("Sign in to download this file.", 401);
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    if (sessionError || !session?.access_token || session.user.id !== user.id) {
      return errorResponse("Sign in to download this file.", 401);
    }

    const upstreamUrl = new URL(path, process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000");
    upstreamUrl.search = incoming.searchParams.get("query") || "";
    const upstream = await fetch(upstreamUrl, {
      headers: { Authorization: `Bearer ${session.access_token}` },
      cache: "no-store",
      redirect: "manual",
      signal: request.signal,
    });
    if (!upstream.ok) {
      const status = upstream.status >= 400 && upstream.status < 500 ? upstream.status : 502;
      return errorResponse(`Export failed (${upstream.status}). Check your access and selected filters.`, status);
    }
    const contentType = upstream.headers.get("content-type") || "";
    const expectedType = path.endsWith(".pdf") ? "application/pdf" : "text/csv";
    if (contentType.split(";")[0].trim().toLowerCase() !== expectedType || !upstream.body) {
      return errorResponse("The export service returned an unexpected file response.", 502);
    }
    const serverName = upstream.headers.get("content-disposition")?.match(/filename="?([^";\r\n]+)"?/i)?.[1];
    const filename = exportFilename(path, incoming.searchParams.get("filename") || serverName);
    return new Response(upstream.body, {
      status: 200,
      headers: {
        ...PRIVATE_HEADERS,
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  } catch {
    return errorResponse("Unable to download the export. Please try again.", 502);
  }
}
