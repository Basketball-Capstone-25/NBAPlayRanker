/** Export paths served by the backend; never accept an arbitrary proxy target. */
export const EXPORT_PATHS = new Set([
  "/data/team-playtypes.csv",
  "/rank-plays/baseline.csv",
  "/pbp/shots.csv",
  "/shots.csv",
  "/metrics/topk-uplift.csv",
  "/export/shotplan.pdf",
  "/export/playtype-viz.pdf",
]);

export function exportFilename(path: string, requested?: string | null): string {
  const fallback = path.split("/").pop() || "download";
  const extension = path.endsWith(".pdf") ? ".pdf" : ".csv";
  const name = (requested || fallback)
    .replace(/[^a-zA-Z0-9._ -]/g, "_")
    .replace(/^\.+/, "")
    .slice(0, 150) || fallback;
  return name.toLowerCase().endsWith(extension) ? name : `${name}${extension}`;
}

export function localExportUrl(url: string, apiBase: string, filename?: string): string {
  const target = new URL(url, apiBase);
  const base = new URL(apiBase);
  if (target.origin !== base.origin || target.username || target.password || !EXPORT_PATHS.has(target.pathname)) {
    throw new Error("Unsupported export URL.");
  }
  const params = new URLSearchParams({ path: target.pathname, query: target.search.slice(1) });
  if (filename) params.set("filename", exportFilename(target.pathname, filename));
  return `/api/exports?${params.toString()}`;
}
