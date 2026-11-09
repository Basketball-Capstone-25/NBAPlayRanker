export type SummaryFilters = { season: string; our: string; opp: string; k: number; w_off: number };
type SummaryScope = SummaryFilters & {
  period: "season"; w_def: number; side: "offense"; season_phase: "regular_season";
};
type SummarySource = {
  filters: SummaryScope; dataset_version: string; aggregation_version: string; limitations: string[];
};
export type SeasonSummary = SummarySource & ({
  status: "ok"; message: null;
  summary: {
    observed_team_ppp: number; modeled_topk_ppp: number; modeled_uplift_ppp: number;
    team_possessions: number; selected_possessions: number; selected_usage_pct: number; k_returned: number;
  };
  play_types: { rank: number; play_type: string; possessions: number; usage_pct: number }[];
} | { status: "no_data"; message: string; summary: null; play_types: [] });

export const summaryKey = (filters: SummaryFilters) =>
  JSON.stringify([filters.season, filters.our, filters.opp, filters.k, filters.w_off]);

const object = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => finite(value) && value >= 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const close = (a: number, b: number) => Math.abs(a - b) < 0.00001;

/** Check the response scope and denominators before showing summary numbers. */
export function parseSeasonSummary(value: unknown, expected: SummaryFilters): SeasonSummary {
  const invalid = (): never => { throw new Error("The season summary is incomplete. Please try again."); };
  if (!object(value) || !object(value.filters)
    || summaryKey(value.filters as SummaryFilters) !== summaryKey(expected)
    || value.filters.period !== "season" || value.filters.side !== "offense"
    || value.filters.season_phase !== "regular_season"
    || !Number.isInteger(value.filters.k) || value.filters.k < 1 || value.filters.k > 10
    || !nonnegative(value.filters.w_off) || value.filters.w_off > 1
    || !nonnegative(value.filters.w_def) || !close(value.filters.w_def, 1 - expected.w_off)
    || !text(value.dataset_version) || !text(value.aggregation_version)
    || !Array.isArray(value.limitations) || !value.limitations.length || !value.limitations.every(text)
    || !Array.isArray(value.play_types)) return invalid();
  if (value.status === "no_data") {
    if (!text(value.message) || value.summary !== null || value.play_types.length !== 0) return invalid();
    return value as SeasonSummary;
  }
  if (value.status !== "ok" || value.message !== null || !object(value.summary)) return invalid();
  const summary = value.summary;
  if (!nonnegative(summary.observed_team_ppp) || !nonnegative(summary.modeled_topk_ppp)
    || !finite(summary.modeled_uplift_ppp)
    || !close(summary.modeled_uplift_ppp, summary.modeled_topk_ppp - summary.observed_team_ppp)
    || !finite(summary.team_possessions) || summary.team_possessions <= 0
    || !finite(summary.selected_possessions) || summary.selected_possessions <= 0
    || summary.selected_possessions > summary.team_possessions
    || !nonnegative(summary.selected_usage_pct) || summary.selected_usage_pct > 100
    || !close(summary.selected_usage_pct, 100 * summary.selected_possessions / summary.team_possessions)
    || !Number.isInteger(summary.k_returned) || summary.k_returned < 1 || summary.k_returned > expected.k
    || summary.k_returned !== value.play_types.length) return invalid();
  for (const [index, row] of value.play_types.entries()) {
    if (!object(row) || row.rank !== index + 1 || !text(row.play_type)
      || !finite(row.possessions) || row.possessions <= 0
      || !nonnegative(row.usage_pct) || row.usage_pct > 100
      || !close(row.usage_pct, 100 * row.possessions / summary.team_possessions)) return invalid();
  }
  if (new Set(value.play_types.map(row => row.play_type)).size !== value.play_types.length
    || !close(value.play_types.reduce((total, row) => total + row.possessions, 0), summary.selected_possessions)
    || !close(value.play_types.reduce((total, row) => total + row.usage_pct, 0), summary.selected_usage_pct)) return invalid();
  return value as SeasonSummary;
}
