export type TeamComparisonFilters = { season: string; team: string; opponent: string };
export type TeamProfile = { possessions: number; points: number; usage_pct: number | null; ppp: number | null };
export type TeamComparison = {
  status: "ok" | "no_data";
  filters: TeamComparisonFilters & { side: "offense"; season_phase: "regular_season" };
  dataset_version: string; aggregation_version: string;
  team_possessions: number | null; opponent_possessions: number | null;
  rows: { play_type: string; team: TeamProfile | null; opponent: TeamProfile | null;
    difference: { usage_pp: number | null; ppp: number | null } }[];
};

export const teamComparisonKey = (filters: TeamComparisonFilters) =>
  JSON.stringify([filters.season, filters.team, filters.opponent]);
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const same = (value: unknown, expected: number | null) => expected === null ? value === null
  : typeof value === "number" && Number.isFinite(value) && Math.abs(value - expected) < 0.00001;

/** Keep each team's full denominator and reject replies for an older selection. */
export function parseTeamComparison(value: unknown, expected: TeamComparisonFilters): TeamComparison {
  const invalid = (): never => { throw new Error("The team comparison is incomplete. Please try again."); };
  if (!object(value) || !object(value.filters) || expected.team === expected.opponent
    || value.filters.season !== expected.season || value.filters.team !== expected.team
    || value.filters.opponent !== expected.opponent || value.filters.side !== "offense"
    || value.filters.season_phase !== "regular_season" || !text(value.dataset_version)
    || !text(value.aggregation_version) || !Array.isArray(value.rows) || value.rows.length > 30) return invalid();
  const totals = { team: value.team_possessions, opponent: value.opponent_possessions };
  for (const total of Object.values(totals)) if (total !== null && !count(total)) return invalid();
  const sums = { team: 0, opponent: 0 }, samples = { team: 0, opponent: 0 };
  const seen = new Set<string>();
  for (const row of value.rows) {
    if (!object(row) || !text(row.play_type) || row.play_type.length > 40
      || seen.has(row.play_type) || !object(row.difference)) return invalid();
    seen.add(row.play_type);
    if (row.team === null && row.opponent === null) return invalid();
    for (const side of ["team", "opponent"] as const) {
      const sample = row[side], total = totals[side];
      if (sample === null) continue;
      if (!object(sample) || !count(total) || !count(sample.possessions) || !count(sample.points)
        || sample.possessions > total || (sample.possessions === 0 && sample.points !== 0)
        || !same(sample.usage_pct, total > 0 ? 100 * sample.possessions / total : null)
        || !same(sample.ppp, sample.possessions > 0 ? sample.points / sample.possessions : null)) return invalid();
      sums[side] += sample.possessions;
      samples[side]++;
    }
    for (const [difference, rate] of [["usage_pp", "usage_pct"], ["ppp", "ppp"]] as const) {
      const left = object(row.team) ? row.team[rate] : null;
      const right = object(row.opponent) ? row.opponent[rate] : null;
      const expectedDifference = typeof left === "number" && typeof right === "number" ? left - right : null;
      if (!same(row.difference[difference], expectedDifference)) return invalid();
    }
  }
  for (const side of ["team", "opponent"] as const) {
    if (totals[side] === null ? samples[side] !== 0 : samples[side] === 0 || sums[side] !== totals[side]) return invalid();
  }
  if (value.status !== (samples.team && samples.opponent ? "ok" : "no_data")) return invalid();
  return value as TeamComparison;
}
