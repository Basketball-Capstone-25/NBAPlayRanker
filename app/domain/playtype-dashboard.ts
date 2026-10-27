export type DashboardFilters = { season: string; our: string; opp: string; min_poss: number };

export type UsageEfficiencyRow = {
  play_type: string; possessions: number; points: number; usage_pct: number; ppp: number;
  league_usage_pct: number; league_ppp: number; usage_delta_pp: number; ppp_delta: number; sample_ok: boolean;
};
export type OpponentDefenseRow = {
  play_type: string; possessions: number; points_allowed: number; allowed_ppp: number;
  league_allowed_ppp: number; allowed_ppp_delta: number; sample_ok: boolean; weak_spot_rank: number | null;
};
export type PlaytypeDashboard = {
  schema_version: "playtype_dashboard_v1";
  aggregation_version: string;
  filters: DashboardFilters & { season_phase: "regular_season" };
  dataset_versions: Record<string, string>;
  coverage: { offense_types: number; defense_types: number; unavailable_defense_types: string[] };
  usage_efficiency: { team: string; total_possessions: number; league_total_possessions: number; rows: UsageEfficiencyRow[] };
  opponent_defense: { team: string; tracked_possessions: number; rows: OpponentDefenseRow[] };
  warnings: string[];
};

export function dashboardKey(filters: DashboardFilters) {
  return JSON.stringify([filters.season, filters.our, filters.opp, filters.min_poss]);
}

export const playtypeName = (name: string) => ({
  PRBallHandler: "Pick and roll handler", PRRollMan: "Pick and roll roller", Spotup: "Spot-up",
  Postup: "Post-up", OffScreen: "Off screen", OffRebound: "Offensive rebound", Misc: "Other",
} as Record<string, string>)[name] || name;

const object = (value: unknown): value is Record<string, any> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const number = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => number(value) && value >= 0;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(v => typeof v === "string");
const close = (a: number, b: number) => Math.abs(a - b) < 0.00001;

/** A mismatched or incomplete response must never become a convincing chart. */
export function parsePlaytypeDashboard(value: unknown, expected: DashboardFilters): PlaytypeDashboard {
  const invalid = () => { throw new Error("The comparison data is incomplete. Please try again."); };
  if (!object(value) || value.schema_version !== "playtype_dashboard_v1"
    || typeof value.aggregation_version !== "string" || !value.aggregation_version
    || !object(value.filters) || dashboardKey(value.filters as DashboardFilters) !== dashboardKey(expected)
    || value.filters.season_phase !== "regular_season" || !object(value.dataset_versions)
    || !Object.keys(value.dataset_versions).length
    || !Object.values(value.dataset_versions).every(v => typeof v === "string" && v.length > 0)
    || !object(value.coverage) || !strings(value.coverage.unavailable_defense_types)
    || !object(value.usage_efficiency) || !object(value.opponent_defense) || !strings(value.warnings)) return invalid();
  const usage = value.usage_efficiency, defense = value.opponent_defense;
  if (usage.team !== expected.our || defense.team !== expected.opp
    || !nonnegative(usage.total_possessions) || !nonnegative(usage.league_total_possessions)
    || !nonnegative(defense.tracked_possessions) || !Array.isArray(usage.rows) || !Array.isArray(defense.rows)
    || usage.rows.length > 30 || defense.rows.length > 30
    || value.coverage.offense_types !== usage.rows.length || value.coverage.defense_types !== defense.rows.length) return invalid();
  const common = (r: Record<string, any>) => typeof r.play_type === "string" && r.play_type.length > 0
    && nonnegative(r.possessions) && r.possessions > 0 && r.sample_ok === (r.possessions >= expected.min_poss);
  for (const row of usage.rows) {
    if (!object(row) || !common(row) || !nonnegative(row.points) || !nonnegative(row.ppp)
      || !nonnegative(row.usage_pct) || row.usage_pct > 100 || !nonnegative(row.league_usage_pct) || row.league_usage_pct > 100
      || !nonnegative(row.league_ppp) || !number(row.usage_delta_pp) || !number(row.ppp_delta)
      || !close(row.ppp, row.points / row.possessions)
      || !close(row.usage_pct, 100 * row.possessions / usage.total_possessions)
      || !close(row.usage_delta_pp, row.usage_pct - row.league_usage_pct)
      || !close(row.ppp_delta, row.ppp - row.league_ppp)) return invalid();
  }
  for (const row of defense.rows) {
    if (!object(row) || !common(row) || !nonnegative(row.points_allowed) || !nonnegative(row.allowed_ppp)
      || !nonnegative(row.league_allowed_ppp) || !number(row.allowed_ppp_delta)
      || !close(row.allowed_ppp, row.points_allowed / row.possessions)
      || !close(row.allowed_ppp_delta, row.allowed_ppp - row.league_allowed_ppp)
      || !(row.weak_spot_rank === null || [1, 2, 3].includes(row.weak_spot_rank))) return invalid();
  }
  if (new Set(usage.rows.map(r => r.play_type)).size !== usage.rows.length
    || new Set(defense.rows.map(r => r.play_type)).size !== defense.rows.length
    || !close(usage.rows.reduce((n, r) => n + r.possessions, 0), usage.total_possessions)
    || !close(defense.rows.reduce((n, r) => n + r.possessions, 0), defense.tracked_possessions)
    || value.coverage.unavailable_defense_types.some(t => defense.rows.some(r => r.play_type === t))) return invalid();
  for (let i = 1; i < defense.rows.length; i++) {
    if (defense.rows[i].allowed_ppp_delta > defense.rows[i - 1].allowed_ppp_delta) return invalid();
  }
  const weakSpots = defense.rows.filter(r => r.sample_ok && r.allowed_ppp_delta > 1e-12).slice(0, 3);
  for (const row of defense.rows) {
    const index = weakSpots.indexOf(row);
    if (row.weak_spot_rank !== (index < 0 ? null : index + 1)) return invalid();
  }
  return value as PlaytypeDashboard;
}
