/** API client: all fetch helpers for the FastAPI backend. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export const FALLBACK_SEASONS = [
  "2019-20",
  "2020-21",
  "2021-22",
  "2022-23",
  "2023-24",
  "2024-25",
];

export const FALLBACK_TEAMS = [
  "ATL","BKN","BOS","CHA","CHI","CLE",
  "DAL","DEN","DET","GSW","HOU","IND",
  "LAC","LAL","MEM","MIA","MIL","MIN",
  "NOP","NYK","OKC","ORL","PHI","PHX",
  "POR","SAC","SAS","TOR","UTA","WAS",
];

// Small fetch helpers
//
// We keep these tiny and predictable.
// - fetchJson(): strict (throws on any non-2xx), good for stable Dataset1 endpoints.
// - fetchJsonWithStatus(): throws an ApiError that includes HTTP status so we can fallback
//   (super useful for Dataset2 because routes might be mounted under /pbp or at root).

import { createClient } from "../../lib/supabase/client";

/** Get Authorization headers for the current Supabase session. */
async function getAuthHeaders(): Promise<Record<string, string>> {
  try {
    const supabase = createClient();
    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    if (token) {
      return { Authorization: `Bearer ${token}` };
    }
  } catch {
    // Supabase not configured — continue without auth
  }
  return {};
}

async function fetchJson<T>(url: string): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(url, { cache: "no-store", headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return (await res.json()) as T;
}

class ApiError extends Error {
  status: number;
  url: string;
  body: string;

  constructor(status: number, url: string, body: string) {
    super(`API error ${status}: ${body}`);
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

async function fetchJsonWithStatus<T>(url: string): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(url, { cache: "no-store", headers });
  const text = await res.text();

  if (!res.ok) {
    throw new ApiError(res.status, url, text);
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    // If backend accidentally returned non-JSON, surface it clearly.
    throw new ApiError(res.status, url, text);
  }
}

/**
 * Authenticated file download (PDF, CSV, etc.).
 * Uses fetch + blob to attach the auth header, then triggers a browser download.
 */
export async function authenticatedDownload(url: string, filename?: string): Promise<void> {
  const headers = await getAuthHeaders();
  const res = await fetch(url, { cache: "no-store", headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Download failed (${res.status}): ${text}`);
  }
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = filename || extractFilename(res) || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(blobUrl);
}

function extractFilename(res: Response): string | null {
  const cd = res.headers.get("Content-Disposition");
  if (!cd) return null;
  const match = cd.match(/filename="?([^";\s]+)"?/);
  return match?.[1] ?? null;
}

/** Try a list of candidate URLs in order and return the first one that works. */
async function tryJsonCandidates<T>(
  urls: string[],
  opts?: {
    // If true, we'll keep trying after 400/422 (useful when param names differ).
    // If false, we'll stop on non-404 errors.
    keepTryingOnClientError?: boolean;
  }
): Promise<{ data: T; usedUrl: string }> {
  let lastErr: unknown = null;

  for (const url of urls) {
    try {
      const data = await fetchJsonWithStatus<T>(url);
      return { data, usedUrl: url };
    } catch (e: any) {
      lastErr = e;

      // 404 => endpoint doesn't exist at this path; try the next candidate.
      if (e?.status === 404) continue;

      // Some endpoints exist but have different query param names, which can cause 422.
      // If keepTryingOnClientError is enabled, continue trying alternates.
      if (opts?.keepTryingOnClientError && (e?.status === 400 || e?.status === 422)) {
        continue;
      }

      // Otherwise stop early to avoid hiding real server errors (500, etc.)
      break;
    }
  }

  const attempted = urls.map((u) => `- ${u}`).join("\n");
  const msg = (lastErr as any)?.message ?? "Unknown error";
  throw new Error(`Request failed.\nTried:\n${attempted}\n\nLast error: ${msg}`);
}

// Types (Dataset1)

export type MetaOptions = {
  seasons: string[];
  teams: string[];
  teamNames?: Record<string, string>;
  playTypes?: string[];
  sides?: string[];
  hasMlPredictions?: boolean;
  _fallback?: boolean;
};

export type TeamPlaytypesPreviewResponse = {
  season: string;
  total_rows: number;
  returned_rows: number;
  rows: Record<string, any>[];
};

export type BaselineRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  w_off: number;
  w_def: number;
  rankings: Record<string, any>[];
};

export type ContextRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  margin: number;
  period: number;
  time_remaining_period_sec: number;
  w_off: number;
  w_def: number;
  rankings: Record<string, any>[];
};

export type ModelMetricsResponse = {
  n_splits: number;
  metrics: Array<{
    model: string;
    RMSE_mean: number;
    RMSE_std: number;
    MAE_mean: number;
    MAE_std: number;
    R2_mean: number;
    R2_std: number;
  }>;
  rf_vs_baseline_t: number | null;
  rf_vs_baseline_p: number | null;
};

// Statistical analysis response (what /analysis/ml returns)
export type MlAnalysisResponse = {
  dataset: any;
  eda: any;
  correlations: { labels: string[]; matrix: number[][] };
  target_feature_corr: Array<{ feature: string; corr: number; abs: number }>;
  feature_selection: any;
  model_selection: any;
};

// Types (Dataset2 / Shots Intelligence + Viz)

// Shot Plan ranking response (Baseline shot recommender on Dataset2)
export type ShotPlanRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  w_off: number;
  w_def: number;

  best_shooter?: any;
  top_shot_types: Record<string, any>[];
  top_zones: Record<string, any>[];
  top_pairs?: Record<string, any>[];

  metadata?: any;
  notes?: string[];

  // Optional debug: which endpoint actually served this response.
  _endpoint_used?: string;
};

// Heatmap responses can come from:
// - Dataset1/root endpoint:     GET /viz/shot-heatmap      -> { caption, image_base64 }
// - Dataset2/PBP endpoint:      GET /pbp/viz/shot-heatmap  -> { season, team, opp, shot_type, zone, max_shots, image_base64 }
export type ShotHeatmapResponse = {
  image_base64: string;
  caption?: string;
  season?: string;
  team?: string;
  opp?: string;
  shot_type?: string | null;
  zone?: string | null;
  max_shots?: number;

  // Optional debug: which endpoint actually served this response.
  _endpoint_used?: string;
};

export type ShotModelMetricsResponse = {
  n_splits: number;
  metrics: Array<{
    model: string;
    RMSE_mean: number;
    RMSE_std: number;
    MAE_mean: number;
    MAE_std: number;
    R2_mean: number;
    R2_std: number;
  }>;
};

export type ShotMlAnalysisResponse = {
  dataset: any;
  eda: any;
  correlations: { labels: string[]; matrix: number[][] };
  target_feature_corr: Array<{ feature: string; corr: number; abs: number }>;
  feature_selection: any;
  model_selection: any;
};

// Dataset2: PBP meta options for dropdowns
export type PbpMetaOptions = {
  seasons: string[];
  teams: string[];
  shotTypes?: string[];
  zones?: string[];
};

// Dataset2: PBP shots preview response for Shots Explorer page
export type PbpShotsPreviewResponse = {
  season: string;
  team: string;
  opp?: string | null;
  shot_type?: string | null;
  zone?: string | null;
  total_rows: number;
  returned_rows: number;
  columns: string[];
  rows: Record<string, any>[];

  // Optional debug: which endpoint actually served this response.
  _endpoint_used?: string;
};

// Helpers to keep UI from crashing
//
// You hit a runtime error like:
//   data.feature_selection.correlation_filter.threshold is undefined
//
// That’s a *shape mismatch* issue. The cleanest fix is to normalize the response
// here so pages can safely read expected fields without tons of optional chaining.

function normalizeFeatureSelection(fs: any): any {
  const out = fs && typeof fs === "object" ? { ...fs } : {};

  const cfRaw = out.correlation_filter && typeof out.correlation_filter === "object"
    ? out.correlation_filter
    : {};

  out.correlation_filter = {
    threshold: cfRaw.threshold ?? null,
    kept: Array.isArray(cfRaw.kept) ? cfRaw.kept : [],
    removed: Array.isArray(cfRaw.removed) ? cfRaw.removed : [],
  };

  // You can extend normalization here if pages rely on other keys.
  return out;
}

function normalizeAnalysisResponse<T extends { feature_selection?: any }>(raw: any): T {
  const obj = raw && typeof raw === "object" ? { ...raw } : {};
  obj.feature_selection = normalizeFeatureSelection(obj.feature_selection);
  return obj as T;
}

// Meta (Dataset1)

export async function fetchMetaOptions(): Promise<MetaOptions> {
  try {
    return await fetchJson<MetaOptions>(`${API_BASE}/meta/options`);
  } catch {
    // If backend isn't running, keep UI usable with fallback dropdowns.
    return {
      seasons: FALLBACK_SEASONS,
      teams: FALLBACK_TEAMS,
      teamNames: {},
      playTypes: [],
      sides: ["offense", "defense"],
      hasMlPredictions: false,
      _fallback: true,
    };
  }
}

// Dataset2 (PBP) Meta
//
// We keep Dataset2 dropdowns separate from Dataset1 meta, because Dataset2 has
// its own set of seasons/teams coming from the PBP shots parquet.
//
// Backend exposes:
//   GET /pbp/meta/options
// returning:
//   { seasons: [...], teams: [...], shotTypes: [...], zones: [...] }
//
// Defensive behavior:
// - If /pbp isn't mounted, we fall back to Dataset1 meta so UI doesn't crash.

export async function fetchPbpMetaOptions(): Promise<PbpMetaOptions> {
  try {
    return await fetchJson<PbpMetaOptions>(`${API_BASE}/pbp/meta/options`);
  } catch {
    const m = await fetchMetaOptions();
    return {
      seasons: m.seasons ?? FALLBACK_SEASONS,
      teams: m.teams ?? FALLBACK_TEAMS,
      shotTypes: [],
      zones: [],
    };
  }
}

export async function fetchPipelineInfo(): Promise<any> {
  return await fetchJson(`${API_BASE}/meta/pipeline`);
}

export async function fetchBaselineInfo(): Promise<any> {
  return await fetchJson(`${API_BASE}/meta/baseline-formula`);
}

// Data Explorer (Dataset1)

export async function fetchTeamPlaytypesPreview(opts: {
  season: string;
  team?: string;
  side?: string;
  playType?: string;
  minPoss?: number;
  limit?: number;
}): Promise<TeamPlaytypesPreviewResponse> {
  const { season, team, side, playType, minPoss = 0, limit = 200 } = opts;

  const params = new URLSearchParams();
  params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  params.set("min_poss", String(minPoss));
  params.set("limit", String(limit));

  return await fetchJson<TeamPlaytypesPreviewResponse>(
    `${API_BASE}/data/team-playtypes?${params.toString()}`
  );
}

export function getTeamPlaytypesCsvUrl(opts: {
  season: string;
  team?: string;
  side?: string;
  playType?: string;
  minPoss?: number;
  limit?: number; //
}): string {
  const { season, team, side, playType, minPoss = 0 } = opts;

  const params = new URLSearchParams();
  params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  params.set("min_poss", String(minPoss));

  return `${API_BASE}/data/team-playtypes.csv?${params.toString()}`;
}

// Baseline ranking (Dataset1)

export async function baselineRank(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  wDef?: number;
}): Promise<
  Array<{
    playType: string;
    pppPred: number;
    pppOff: number;
    pppDef: number;
    pppGap: number;
    rationale: string;
    raw: Record<string, any>;
  }>
> {
  // ✅ FIX: destructure wDef so it exists (and give it a safe default)
  const { season, our, opp, k = 5, wOff = 0.7, wDef = 0.3 } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(wDef),
  });

  const data = await fetchJson<BaselineRankResponse>(
    `${API_BASE}/rank-plays/baseline?${params.toString()}`
  );

  const rankings = Array.isArray(data.rankings) ? data.rankings : [];

  return rankings.map((r) => ({
    playType: r.PLAY_TYPE,
    pppPred: Number(r.PPP_PRED),
    pppOff: Number(r.PPP_OFF_SHRUNK),
    pppDef: Number(r.PPP_DEF_SHRUNK),
    pppGap: Number(r.PPP_GAP),
    rationale: r.RATIONALE || "",
    raw: r,
  }));
}

export function getBaselineCsvUrl(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  wDef?: number;
}): string {
  const { season, our, opp, k = 5, wOff = 0.7 } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(opts.wDef ?? 0.3),
  });

  return `${API_BASE}/rank-plays/baseline.csv?${params.toString()}`;
}

// Context + ML ranking (Dataset1)

export async function contextRank(opts: {
  season: string;
  our: string;
  opp: string;
  margin: number;
  period: number;
  timeRemaining: number; // seconds remaining in current period
  k?: number;
  wOff?: number;
}): Promise<
  Array<{
    playType: string;
    finalPPP: number;
    mlPPP: number;
    baselinePPP: number;
    deltaPPP: number;
    contextLabel: string;
    rationale: string;
    raw: Record<string, any>;
  }>
> {
  const {
    season,
    our,
    opp,
    margin,
    period,
    timeRemaining,
    k = 5,
    wOff = 0.7,
  } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    margin: String(margin),
    period: String(period),
    time_remaining: String(timeRemaining),
    k: String(k),
    w_off: String(wOff),
  });

  const data = await fetchJson<ContextRankResponse>(
    `${API_BASE}/rank-plays/context-ml?${params.toString()}`
  );

  const rows = Array.isArray(data.rankings) ? data.rankings : [];

  return rows.map((r) => ({
    playType: r.PLAY_TYPE,
    finalPPP: Number(r.PPP_CONTEXT),
    mlPPP: Number(r.PPP_ML_BLEND),
    baselinePPP: Number(r.PPP_BASELINE),
    deltaPPP: Number(r.DELTA_VS_BASELINE),
    contextLabel: r.CONTEXT_LABEL || "",
    rationale: r.RATIONALE || "",
    raw: r,
  }));
}

// Model metrics (Dataset1)

export async function fetchModelMetrics(nSplits = 5): Promise<ModelMetricsResponse> {
  const params = new URLSearchParams({ n_splits: String(nSplits) });
  return await fetchJson<ModelMetricsResponse>(
    `${API_BASE}/metrics/baseline-vs-ml?${params.toString()}`
  );
}

// Statistical Analysis (Dataset1)

export async function fetchMlAnalysis(opts?: {
  nSplits?: number;
  minPoss?: number;
  refresh?: boolean;
}): Promise<MlAnalysisResponse> {
  const nSplits = opts?.nSplits ?? 5;
  const minPoss = opts?.minPoss ?? 25;

  const params = new URLSearchParams();
  params.set("n_splits", String(nSplits));
  params.set("min_poss", String(minPoss));
  if (opts?.refresh) params.set("refresh", "true");

  const raw = await fetchJson<MlAnalysisResponse>(
    `${API_BASE}/analysis/ml?${params.toString()}`
  );

  // Normalize so pages can read expected fields safely.
  return normalizeAnalysisResponse<MlAnalysisResponse>(raw);
}

// SportyPy Visualization (Dataset1)

export async function fetchPlaytypeViz(opts: {
  season: string;
  our: string;
  opp: string;
  playType: string;
  wOff: number;
}) {
  const params = new URLSearchParams({
    season: opts.season,
    our: opts.our,
    opp: opts.opp,
    play_type: opts.playType,
    w_off: String(opts.wOff),
  });

  return await fetchJson<{ caption: string; image_base64: string }>(
    `${API_BASE}/viz/playtype-zones?${params.toString()}`
  );
}

// Shot Intelligence (Dataset2)
//

//
// we implement defensive helpers that try the most likely candidate paths
// and also try alternate query param naming (our/opp vs our_team/opp_team).

export async function fetchShotPlanRank(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
}): Promise<ShotPlanRankResponse> {
  const { season, our, opp, k = 5, wOff = 0.7 } = opts;

  // Build multiple param variants (FastAPI ignores unknown params, so this is safe).
  const pA = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
  });

  const pB = new URLSearchParams({
    season,
    our_team: our,
    opp_team: opp,
    k: String(k),
    w_off: String(wOff),
  });

  // Candidate URLs in priority order.
  const candidates = [
    // Preferred Dataset2 mount
    `${API_BASE}/pbp/shotplan/rank?${pA.toString()}`,
    `${API_BASE}/pbp/shotplan/rank?${pB.toString()}`,

    // Legacy / root mounts (some builds do this)
    `${API_BASE}/shotplan/rank?${pA.toString()}`,
    `${API_BASE}/shotplan/rank?${pB.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<ShotPlanRankResponse>(candidates, {
    keepTryingOnClientError: true, // helps if one variant 422s but another works
  });

  // Normalize fields so UI can rely on a consistent shape.
  const safe: ShotPlanRankResponse = {
    season: (data as any)?.season ?? season,
    our_team: (data as any)?.our_team ?? (data as any)?.our ?? our,
    opp_team: (data as any)?.opp_team ?? (data as any)?.opp ?? opp,
    k: (data as any)?.k ?? k,
    w_off: (data as any)?.w_off ?? (data as any)?.wOff ?? wOff,
    w_def: (data as any)?.w_def ?? (data as any)?.wDef ?? (1 - wOff),

    top_shot_types: Array.isArray((data as any)?.top_shot_types)
      ? (data as any).top_shot_types
      : [],
    top_zones: Array.isArray((data as any)?.top_zones)
      ? (data as any).top_zones
      : [],
    top_pairs: Array.isArray((data as any)?.top_pairs)
      ? (data as any).top_pairs
      : undefined,

    best_shooter: (data as any)?.best_shooter,
    metadata: (data as any)?.metadata,
    notes: Array.isArray((data as any)?.notes) ? (data as any).notes : undefined,
    _endpoint_used: usedUrl,
  };

  return safe;
}

export async function fetchShotHeatmap(opts: {
  season: string;
  team?: string;
  our?: string;
  opp: string;
  shotType?: string;
  zone?: string;
  maxShots?: number;
}): Promise<ShotHeatmapResponse> {
  const team = opts.team ?? opts.our;
  if (!team) {
    throw new Error("fetchShotHeatmap requires either `team` or `our`.");
  }

  // Dataset2 query style (team/opp) + max_shots
  const pPbp = new URLSearchParams({
    season: opts.season,
    team,
    opp: opts.opp,
  });
  if (opts.shotType) pPbp.set("shot_type", opts.shotType);
  if (opts.zone) pPbp.set("zone", opts.zone);
  if (opts.maxShots != null) pPbp.set("max_shots", String(opts.maxShots));

  // Legacy/root query style (our/opp)
  const pRoot = new URLSearchParams({
    season: opts.season,
    our: team,
    opp: opts.opp,
  });
  if (opts.shotType) pRoot.set("shot_type", opts.shotType);
  if (opts.zone) pRoot.set("zone", opts.zone);

  const candidates = [
    // Preferred Dataset2 name
    `${API_BASE}/pbp/viz/shot-heatmap?${pPbp.toString()}`,

    // Some people accidentally name it /pbp/viz/heatmap ( earlier URL)
    `${API_BASE}/pbp/viz/heatmap?${pPbp.toString()}`,

    // Legacy/root endpoint
    `${API_BASE}/viz/shot-heatmap?${pRoot.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<ShotHeatmapResponse>(candidates, {
    keepTryingOnClientError: true,
  });

  // Normalize so UI always gets image_base64
  if (!(data as any)?.image_base64) {
    throw new Error(
      `Heatmap response missing image_base64. Endpoint used: ${usedUrl}`
    );
  }

  return {
    ...data,
    _endpoint_used: usedUrl,
  };
}

export function getShotPlanPdfUrl(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  shotType?: string;
  zone?: string;
  maxShots?: number;
}): string {
  // Keep PDF export on the legacy root endpoint unless backend explicitly adds /pbp/export.
  // This keeps existing exports stable.
  const { season, our, opp, k = 5, wOff = 0.7, shotType, zone } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
  });

  if (shotType) params.set("shot_type", shotType);
  if (zone) params.set("zone", zone);

  return `${API_BASE}/export/shotplan.pdf?${params.toString()}`;
}

export async function fetchShotMlAnalysis(opts?: {
  nSplits?: number;
  refresh?: boolean;
}): Promise<ShotMlAnalysisResponse> {
  const nSplits = opts?.nSplits ?? 5;
  const params = new URLSearchParams();
  params.set("n_splits", String(nSplits));
  if (opts?.refresh) params.set("refresh", "true");

  const raw = await fetchJson<ShotMlAnalysisResponse>(
    `${API_BASE}/analysis/shot-ml?${params.toString()}`
  );

  // Normalize so pages can safely access correlation_filter.threshold, kept, etc.
  return normalizeAnalysisResponse<ShotMlAnalysisResponse>(raw);
}

export async function fetchShotModelMetrics(
  nSplits = 5
): Promise<ShotModelMetricsResponse> {
  const params = new URLSearchParams({ n_splits: String(nSplits) });
  return await fetchJson<ShotModelMetricsResponse>(
    `${API_BASE}/metrics/shot-models?${params.toString()}`
  );
}

// Dataset2 Shots Explorer helpers (WHAT shot-explorer PAGE NEEDS)
//
//  backend currently 404s on:
//   GET /pbp/shots/preview
//
// That means either:
// - the endpoint isn't implemented, OR
// - it exists at a different mount path (often /shots/preview)
//
// We handle that here by trying both paths so the UI doesn't break.

function buildPbpShotsParams(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}) {
  const params = new URLSearchParams();

  params.set("season", opts.season);
  params.set("limit", String(opts.limit ?? 50));

  // Team naming variants: some endpoints use team, some use our
  params.set("team", opts.team);
  params.set("our", opts.team);

  if (opts.opp) params.set("opp", opts.opp);

  // Shot type naming variants
  if (opts.shotType) {
    params.set("shot_type", opts.shotType);
    params.set("shotType", opts.shotType);
  }

  if (opts.zone) params.set("zone", opts.zone);

  return params;
}

export async function fetchPbpShotsPreview(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): Promise<PbpShotsPreviewResponse> {
  const params = buildPbpShotsParams(opts);

  const candidates = [
    // Preferred Dataset2 mount
    `${API_BASE}/pbp/shots/preview?${params.toString()}`,

    // Fallback if backend mounted it at root
    `${API_BASE}/shots/preview?${params.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<PbpShotsPreviewResponse>(candidates, {
    keepTryingOnClientError: true,
  });

  // Normalize table-safe shape (so UI never crashes)
  const safe: PbpShotsPreviewResponse = {
    season: (data as any)?.season ?? opts.season,
    team: (data as any)?.team ?? (data as any)?.our ?? opts.team,
    opp: (data as any)?.opp ?? null,
    shot_type: (data as any)?.shot_type ?? null,
    zone: (data as any)?.zone ?? null,

    total_rows: Number((data as any)?.total_rows ?? 0),
    returned_rows: Number((data as any)?.returned_rows ?? (data as any)?.rows?.length ?? 0),

    columns: Array.isArray((data as any)?.columns) ? (data as any).columns : [],
    rows: Array.isArray((data as any)?.rows) ? (data as any).rows : [],

    _endpoint_used: usedUrl,
  };

  return safe;
}

export function getPbpShotsCsvUrl(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): string {
  const params = buildPbpShotsParams({
    ...opts,
    limit: opts.limit ?? 5000, // exports usually larger than preview
  });

  // Prefer /pbp path
  return `${API_BASE}/pbp/shots.csv?${params.toString()}`;
}

/** Optional helper: if /pbp/shots.csv is not mounted in  backend, */
export function getPbpShotsCsvUrlLegacy(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): string {
  const params = buildPbpShotsParams({
    ...opts,
    limit: opts.limit ?? 5000,
  });

  return `${API_BASE}/shots.csv?${params.toString()}`;
}
