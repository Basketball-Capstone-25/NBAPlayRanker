export type LeagueFilters = { season: string; team: string; side: "offense" | "defense" };
export type LeagueRate = "fg_pct" | "efg_pct" | "tov_pct" | "ft_pct";
export type LeagueSample = Record<LeagueRate, number | null> & {
  possessions: number; field_goal_attempts: number;
};
export type LeagueComparison = {
  status: "ok" | "no_data";
  filters: LeagueFilters & { season_phase: "regular_season" };
  selected_team_in_league: true;
  dataset_version: string; aggregation_version: string;
  rows: {
    play_type: string; team: LeagueSample | null; league: LeagueSample | null;
    difference_pp: Record<LeagueRate, number | null>;
  }[];
};

export const leagueKey = (filters: LeagueFilters) => JSON.stringify([filters.season, filters.team, filters.side]);
const rates: LeagueRate[] = ["fg_pct", "efg_pct", "tov_pct", "ft_pct"];
const object = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const count = (value: unknown): value is number => finite(value) && Number.isSafeInteger(value) && value >= 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

function validSample(value: unknown): value is LeagueSample | null {
  if (value === null) return true;
  if (!object(value) || !count(value.possessions) || !count(value.field_goal_attempts)) return false;
  return rates.every(rate => {
    const denominator = rate === "fg_pct" || rate === "efg_pct" ? value.field_goal_attempts : value.possessions;
    return denominator === 0 ? value[rate] === null
      : finite(value[rate]) && value[rate] >= 0 && value[rate] <= (rate === "efg_pct" ? 1.5 : 1);
  });
}

/** Keep the selected scope, sample counts and percentage-point arithmetic together. */
export function parseLeagueComparison(value: unknown, expected: LeagueFilters): LeagueComparison {
  const invalid = (): never => { throw new Error("The league comparison is incomplete. Please try again."); };
  if (!object(value) || !object(value.filters)
    || leagueKey(value.filters as LeagueFilters) !== leagueKey(expected)
    || !["offense", "defense"].includes(value.filters.side)
    || value.filters.season_phase !== "regular_season" || value.selected_team_in_league !== true
    || !text(value.dataset_version) || !text(value.aggregation_version) || !Array.isArray(value.rows)
    || value.rows.length > 30) return invalid();
  if (value.status === "no_data") {
    if (value.rows.length) return invalid();
    return value as LeagueComparison;
  }
  if (value.status !== "ok" || !value.rows.length) return invalid();
  const seen = new Set<string>();
  for (const row of value.rows) {
    if (!object(row) || !text(row.play_type) || seen.has(row.play_type)
      || !validSample(row.team) || !validSample(row.league) || (!row.team && !row.league)
      || !object(row.difference_pp)) return invalid();
    seen.add(row.play_type);
    for (const rate of rates) {
      const own = row.team?.[rate], league = row.league?.[rate], delta = row.difference_pp[rate];
      if (own == null || league == null) {
        if (delta !== null) return invalid();
      } else if (!finite(delta) || Math.abs(delta - 100 * (own - league)) > 0.00001) return invalid();
    }
  }
  if (!value.rows.some(row => row.team !== null)) return invalid();
  return value as LeagueComparison;
}
