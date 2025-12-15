// app/utils.js
//
// Frontend API helpers for the PSPI.
//
// Goals:
// - Keep the UI thin: pages call these helpers instead of embedding fetch logic.
// - Make it obvious which endpoints support which use cases.
// - Use backend-provided dropdown options when available (so we don’t hardcode seasons/teams).
//
// Endpoints used:
// - GET  /meta/options
// - GET  /meta/pipeline
// - GET  /meta/baseline
// - GET  /data/team-playtypes
// - GET  /data/team-playtypes.csv
// - GET  /rank-plays/baseline
// - GET  /rank-plays/baseline.csv
// - GET  /rank-plays/context-ml
// - GET  /metrics/baseline-vs-ml

// Base URL for the FastAPI backend
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

/**
 * Fallback lists (used only if /meta/options is unreachable).
 * Keeping these avoids blank dropdowns during local dev if backend isn't running.
 */
export const FALLBACK_SEASONS = [
  "2019-20",
  "2020-21",
  "2021-22",
  "2022-23",
  "2023-24",
  "2024-25",
];

export const FALLBACK_TEAMS = [
  "ATL", "BKN", "BOS", "CHA", "CHI", "CLE",
  "DAL", "DEN", "DET", "GSW", "HOU", "IND",
  "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
  "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
  "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
];

// ---------------------------
// Small fetch helper
// ---------------------------

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

// ---------------------------
// Meta endpoints (dropdowns / explanations)
// ---------------------------

/**
 * Fetch UI dropdown options dynamically from the dataset.
 * Returns:
 * {
 *   seasons: string[],
 *   teams: string[],
 *   teamNames: { [abbr]: fullName },
 *   playTypes: string[],
 *   sides: ["offense","defense"],
 *   hasMlPredictions: boolean
 * }
 */
export async function fetchMetaOptions() {
  try {
    return await fetchJson(`${API_BASE}/meta/options`);
  } catch (e) {
    // Fallback so UI stays usable if backend is down.
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

/**
 * Fetch pipeline summary (used by Data Explorer + Model Metrics pages).
 * This helps the committee see the cleaning/aggregation steps clearly.
 */
export async function fetchPipelineInfo() {
  return await fetchJson(`${API_BASE}/meta/pipeline`);
}

/**
 * Fetch the baseline formula explanation (used by Matchup/Baseline page).
 */
export async function fetchBaselineInfo() {
  return await fetchJson(`${API_BASE}/meta/baseline`);
}

// ---------------------------
// Data Explorer endpoints
// ---------------------------

/**
 * Preview the cleaned TEAM-LEVEL dataset used by models.
 * IMPORTANT: This is NOT recommendations and NOT predictions.
 *
 * Params:
 * - season, team, side, playType (optional)
 * - minPoss (default 0)
 * - limit (default 200)
 *
 * Response:
 * { total_rows, returned_rows, rows: [...] }
 */
export async function fetchTeamPlaytypesPreview({
  season,
  team,
  side,
  playType,
  minPoss = 0,
  limit = 200,
} = {}) {
  const params = new URLSearchParams();
  if (season) params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  if (minPoss != null) params.set("min_poss", String(minPoss));
  if (limit != null) params.set("limit", String(limit));

  return await fetchJson(`${API_BASE}/data/team-playtypes?${params.toString()}`);
}

/**
 * Build a CSV download URL for the Data Explorer table (filtered export).
 * Use this in <a href=... download>Export CSV</a>
 */
export function getTeamPlaytypesCsvUrl({
  season,
  team,
  side,
  playType,
  minPoss = 0,
  limit = 5000,
} = {}) {
  const params = new URLSearchParams();
  if (season) params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  if (minPoss != null) params.set("min_poss", String(minPoss));
  if (limit != null) params.set("limit", String(limit));

  return `${API_BASE}/data/team-playtypes.csv?${params.toString()}`;
}

// ---------------------------
// Baseline (Explainable) recommendations
// ---------------------------

/**
 * Call the FastAPI baseline ranking endpoint and normalize into a shape pages can use.
 *
 * Backend returns: { rankings: [{ PLAY_TYPE, PPP_PRED, PPP_OFF_SHRUNK, PPP_DEF_SHRUNK, PPP_GAP, RATIONALE, ... }] }
 *
 * Returns: [{ playType, pppPred, pppOff, pppDef, pppGap, rationale, raw }]
 */
export async function baselineRank({
  season,
  our,
  opp,
  k = 5,
  wOff = 0.7,
  wDef = 0.3,
}) {
  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(wDef),
  });

  const data = await fetchJson(`${API_BASE}/rank-plays/baseline?${params.toString()}`);
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

/**
 * CSV download URL for baseline recommendations
 */
export function getBaselineCsvUrl({ season, our, opp, k = 5, wOff = 0.7, wDef = 0.3 }) {
  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(wDef),
  });
  return `${API_BASE}/rank-plays/baseline.csv?${params.toString()}`;
}

// ---------------------------
// Context + ML recommendations (AI use case)
// ---------------------------

/**
 * Call the context+ML endpoint.
 *
 * Backend returns: { rankings: [{ PLAY_TYPE, PPP_CONTEXT, PPP_ML_BLEND, PPP_BASELINE, DELTA_VS_BASELINE, CONTEXT_LABEL, RATIONALE, ...}] }
 *
 * Returns: [{ playType, finalPPP, mlPPP, baselinePPP, deltaPPP, contextLabel, rationale, raw }]
 */
export async function contextRank({
  season,
  our,
  opp,
  margin,
  period,
  timeRemaining, // seconds remaining in the current period
  k = 5,
}) {
  const params = new URLSearchParams({
    season,
    our,
    opp,
    margin: String(margin),
    period: String(period),
    time_remaining: String(timeRemaining),
    k: String(k),
  });

  const data = await fetchJson(`${API_BASE}/rank-plays/context-ml?${params.toString()}`);
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

// ---------------------------
// Model Metrics (defend the ML model)
// ---------------------------

/**
 * Fetch baseline vs ML evaluation metrics (season holdout).
 *
 * Response shape:
 * {
 *   n_splits: number,
 *   metrics: [
 *     { model, RMSE_mean, RMSE_std, MAE_mean, MAE_std, R2_mean, R2_std }, ...
 *   ],
 *   rf_vs_baseline_t: number|null,
 *   rf_vs_baseline_p: number|null
 * }
 */
export async function fetchModelMetrics(nSplits = 5) {
  const params = new URLSearchParams({ n_splits: String(nSplits) });
  return await fetchJson(`${API_BASE}/metrics/baseline-vs-ml?${params.toString()}`);
}
