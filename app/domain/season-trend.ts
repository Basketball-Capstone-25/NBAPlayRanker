export type TrendFilters = { team: string; play_type: string };
export type TrendRow = {
  season: string; possessions: number | null; points: number | null;
  team_possessions: number | null; usage_pct: number | null; ppp: number | null;
};
export type SeasonTrend = {
  status: "ok" | "no_data";
  filters: TrendFilters & { side: "offense"; season_phase: "regular_season" };
  dataset_version: string; aggregation_version: string; rows: TrendRow[];
};

export const trendKey = (filters: TrendFilters) => JSON.stringify([filters.team, filters.play_type]);
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const same = (value: unknown, expected: number | null) => expected === null ? value === null
  : typeof value === "number" && Number.isFinite(value) && Math.abs(value - expected) < 0.00001;

/** Check the selected scope, consecutive seasons and source-count arithmetic. */
export function parseSeasonTrend(value: unknown, expected: TrendFilters): SeasonTrend {
  const invalid = (): never => { throw new Error("The season trend is incomplete. Please try again."); };
  if (!object(value) || !object(value.filters)
    || value.filters.team !== expected.team || value.filters.play_type !== expected.play_type
    || value.filters.side !== "offense" || value.filters.season_phase !== "regular_season"
    || !text(value.dataset_version) || !text(value.aggregation_version)
    || !Array.isArray(value.rows) || !value.rows.length || value.rows.length > 100) return invalid();
  let previous: number | null = null;
  let hasSample = false;
  for (const row of value.rows) {
    if (!object(row) || typeof row.season !== "string" || !/^20\d{2}-\d{2}$/.test(row.season)) return invalid();
    const year = Number(row.season.slice(0, 4));
    if (row.season.slice(-2) !== String((year + 1) % 100).padStart(2, "0")
      || (previous !== null && year !== previous + 1)) return invalid();
    previous = year;
    const total = row.team_possessions, possessions = row.possessions, points = row.points;
    if (total !== null && !count(total)) return invalid();
    if (possessions === null) {
      if (points !== null || row.usage_pct !== null || row.ppp !== null) return invalid();
      continue;
    }
    if (!count(possessions) || !count(points) || !count(total) || possessions > total
      || (possessions === 0 && points !== 0)
      || !same(row.usage_pct, total > 0 ? 100 * possessions / total : null)
      || !same(row.ppp, possessions > 0 ? points / possessions : null)) return invalid();
    hasSample ||= possessions > 0;
  }
  if (value.status !== (hasSample ? "ok" : "no_data")) return invalid();
  return value as SeasonTrend;
}
