export type OutcomeFilters = { season: string; team: string };
export const OUTCOME_METRICS = [
  { key: "scoring", label: "Scoring", source: "SCORE_POSS_PCT", definition: "Share of possessions that produced points." },
  { key: "turnover", label: "Turnovers", source: "TOV_POSS_PCT", definition: "Share of possessions that ended in a turnover." },
  { key: "free_throw", label: "Free throws", source: "FT_POSS_PCT", definition: "Share of possessions ending with free throws, not free-throw shooting accuracy." },
  { key: "shooting_foul", label: "Shooting fouls", source: "SF_POSS_PCT", definition: "Share of possessions ending with a shooting foul." },
  { key: "and_one", label: "And-ones", source: "PLUSONE_POSS_PCT", definition: "Share of possessions with a made basket and an additional free throw." },
] as const;
export type OutcomeKey = typeof OUTCOME_METRICS[number]["key"];
export type OutcomeRate = {
  value: number | null; status: "ok" | "missing" | "invalid" | "no_possessions";
  usable_possessions: number; missing_possessions: number; invalid_possessions: number;
};
export type PossessionOutcomes = {
  status: "ok" | "no_data";
  filters: OutcomeFilters & { side: "offense"; season_phase: "regular_season" };
  dataset_version: string; aggregation_version: string;
  rows: { play_type: string; possessions: number; rates: Record<OutcomeKey, OutcomeRate> }[];
};
export const outcomeKey = (filters: OutcomeFilters) => JSON.stringify([filters.season, filters.team]);
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

function validRate(value: unknown, possessions: number): value is OutcomeRate {
  if (!object(value) || !count(value.usable_possessions) || !count(value.missing_possessions)
    || !count(value.invalid_possessions)) return false;
  const total = value.usable_possessions + value.missing_possessions + value.invalid_possessions;
  if (!Number.isSafeInteger(total) || total !== possessions) return false;
  const status = value.invalid_possessions > 0 ? "invalid" : value.missing_possessions > 0 ? "missing"
    : possessions > 0 ? "ok" : "no_possessions";
  if (value.status !== status) return false;
  return status === "ok" ? typeof value.value === "number" && Number.isFinite(value.value)
    && value.value >= 0 && value.value <= 1 : value.value === null;
}

/** Rates can overlap; validate each rate and its source coverage independently. */
export function parsePossessionOutcomes(value: unknown, expected: OutcomeFilters): PossessionOutcomes {
  const invalid = (): never => { throw new Error("The possession rates are incomplete. Please try again."); };
  if (!object(value) || !object(value.filters) || value.filters.season !== expected.season
    || value.filters.team !== expected.team || value.filters.side !== "offense"
    || value.filters.season_phase !== "regular_season" || !text(value.dataset_version)
    || !text(value.aggregation_version) || !Array.isArray(value.rows) || value.rows.length > 30
    || value.status !== (value.rows.length ? "ok" : "no_data")) return invalid();
  const seen = new Set<string>();
  for (const row of value.rows) {
    if (!object(row) || !text(row.play_type) || row.play_type.length > 40 || seen.has(row.play_type)
      || !count(row.possessions) || !object(row.rates)) return invalid();
    seen.add(row.play_type);
    for (const { key } of OUTCOME_METRICS) if (!validRate(row.rates[key], row.possessions)) return invalid();
  }
  return value as PossessionOutcomes;
}
