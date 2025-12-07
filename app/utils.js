// app/utils.js

// Base URL for the FastAPI backend
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

// Seasons covered by the Synergy CSV (adjust if needed)
export const SEASONS = [
  "2019-20",
  "2020-21",
  "2021-22",
  "2022-23",
  "2023-24",
  "2024-25",
];

// NBA team abbreviations used in the Synergy file
export const TEAMS = [
  "ATL", "BKN", "BOS", "CHA", "CHI", "CLE",
  "DAL", "DEN", "DET", "GSW", "HOU", "IND",
  "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
  "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
  "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
];

/**
 * Call the FastAPI baseline ranking endpoint and normalise
 * into a shape the React pages can use.
 *
 * Returns: [{ playType, PPP_pred, PPP_off, PPP_def, PPP_gap, raw }]
 */
export async function baselineRank({ season, our, opp, k = 5 }) {
  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
  });

  const url = `${API_BASE}/rank-plays/baseline?${params.toString()}`;

  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Baseline API error ${res.status}: ${text}`);
  }

  const data = await res.json();
  const rankings = Array.isArray(data.rankings) ? data.rankings : [];

  return rankings.map((r) => ({
    playType: r.PLAY_TYPE,
    PPP_pred: r.PPP_PRED,
    PPP_off: r.PPP_OFF_SHRUNK,
    PPP_def: r.PPP_DEF_SHRUNK,
    PPP_gap: r.PPP_GAP,
    raw: r,
  }));
}

export async function contextRank({
  season,
  our,
  opp,
  margin,
  period,
  timeRemaining,
  k = 3,
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

  const res = await fetch(`${API_BASE}/rank-plays/context-ml?${params.toString()}`);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Context ML API error (${res.status}): ${text}`);
  }

  const data = await res.json();

  // Map backend fields to the shape the ContextSimulator will use
  return (data.rankings ?? []).map((r) => {
    const base = Number(r.PPP_PRED_ML);
    const ctx = Number(r.CONTEXT_SCORE);
    return {
      playType: r.PLAY_TYPE,
      basePPP: base,
      contextPPP: ctx,
      deltaPPP: ctx - base,
      rationale: r.CONTEXT_RATIONALE,
    };
  });
}


/**
 * Fetch offline baseline vs ML metrics from the backend.
 *
 * Response shape (from FastAPI):
 * {
 *   n_splits: number,
 *   metrics: [
 *     {
 *       model: string,
 *       RMSE_mean: number, RMSE_std: number,
 *       MAE_mean: number, MAE_std: number,
 *       R2_mean: number,  R2_std: number
 *     }, ...
 *   ]
 * }
 */
export async function fetchModelMetrics(nSplits = 5) {
  const params = new URLSearchParams({
    n_splits: String(nSplits),
  });
  const url = `${API_BASE}/metrics/baseline-vs-ml?${params.toString()}`;

  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Metrics API error ${res.status}: ${text}`);
  }

  const data = await res.json();
  return data;
}
