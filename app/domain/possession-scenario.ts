import type { UsageEfficiencyRow } from "./playtype-dashboard";

/** Reallocate recorded possessions while holding both observed scoring rates fixed. */
export function possessionScenario(from: UsageEfficiencyRow, to: UsageEfficiencyRow, possessions: number) {
  if (from.play_type === to.play_type || !from.sample_ok || !to.sample_ok
    || !Number.isFinite(possessions) || possessions <= 0 || possessions > from.usage_pct
    || !Number.isFinite(from.ppp) || !Number.isFinite(to.ppp) || from.ppp < 0 || to.ppp < 0
    || !Number.isFinite(from.usage_pct) || from.usage_pct < 0 || from.usage_pct > 100) return null;
  return { pointsBefore: possessions * from.ppp, pointsAfter: possessions * to.ppp,
    difference: possessions * (to.ppp - from.ppp) };
}
