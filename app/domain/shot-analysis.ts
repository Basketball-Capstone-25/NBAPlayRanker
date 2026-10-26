export type ShotFeatureScore = { feature: string; score: number | null };
export type ShotAnalysis = {
  dataset: {
    rows_used: number | null;
    n_games: number | null;
    n_seasons: number | null;
    feature_spec: { numeric: string[]; categorical: string[]; target: string | null };
    notes: string[];
  };
  eda: {
    points: { hist: { bins: number[]; counts: number[] } };
    missing_counts: Record<string, number>;
  };
  correlations: { labels: string[]; matrix: (number | null)[][] };
  feature_target_corr: { feature: string; corr: number | null; abs: number | null }[];
  feature_selection: { method: string | null; k: number | null; scores: ShotFeatureScore[] } | null;
};

const object = (x: unknown): Record<string, unknown> => x !== null && typeof x === "object" && !Array.isArray(x) ? x as Record<string, unknown> : {};
const number = (x: unknown) => typeof x === "number" && Number.isFinite(x) ? x : null;
const strings = (x: unknown): string[] => Array.isArray(x) ? x.filter((v): v is string => typeof v === "string") : [];

export function normalizeShotAnalysis(response: unknown): ShotAnalysis {
  const outer = object(response);
  const raw = outer.payload ? object(outer.payload) : outer;
  const dataset = object(raw.dataset), spec = object(dataset.feature_spec);
  const eda = object(raw.eda), hist = object(object(eda.points).hist);
  const correlations = object(raw.correlations), selection = object(raw.feature_selection);
  return {
    dataset: {
      rows_used: number(dataset.rows_used ?? dataset.rows), n_games: number(dataset.n_games), n_seasons: number(dataset.n_seasons),
      feature_spec: { numeric: strings(spec.numeric), categorical: strings(spec.categorical), target: typeof spec.target === "string" ? spec.target : null },
      notes: strings(dataset.notes),
    },
    eda: {
      points: { hist: { bins: Array.isArray(hist.bins) ? hist.bins.filter((x): x is number => number(x) !== null) : [], counts: Array.isArray(hist.counts) ? hist.counts.filter((x): x is number => number(x) !== null) : [] } },
      missing_counts: Object.fromEntries(Object.entries(object(eda.missing_counts)).filter(([, value]) => number(value) !== null)) as Record<string, number>,
    },
    correlations: { labels: strings(correlations.labels), matrix: Array.isArray(correlations.matrix) ? correlations.matrix.map(row => Array.isArray(row) ? row.map(number) : []) : [] },
    feature_target_corr: Array.isArray(raw.feature_target_corr) ? raw.feature_target_corr.map(row => object(row)).filter(row => typeof row.feature === "string").map(row => ({ feature: row.feature as string, corr: number(row.corr), abs: number(row.abs) })) : [],
    feature_selection: typeof selection.method === "string" ? {
      method: selection.method, k: number(selection.k), scores: Array.isArray(selection.scores) ? selection.scores.map(object).filter(row => typeof row.feature === "string").map(row => ({ feature: row.feature as string, score: number(row.score) })) : [],
    } : null,
  };
}

export type ShotLaterSeason = {
  train_seasons: string[];
  test_season: string;
  train_rows: number;
  test_rows: number;
  metrics: { model: string; RMSE: number | null; MAE: number | null; R2: number | null }[];
  limitation: string;
};
