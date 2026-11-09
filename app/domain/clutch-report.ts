// Domain Layer — late-game (clutch) report (SCRUM-629).
// Same limits as backend/application/analytics_services/clutch_report.py so the
// panel can explain a bad filter before calling the API.
import { clampSituation, formatClock, type QuickCallSituation } from "./quick-call";

export type ClutchFilters = {
  season: string;
  team: string;
  opp: string;
  periodFrom: number; // 4 = fourth quarter; overtime is always included
  secondsLeft: number; // seconds left in the period
  marginMin: number; // shooting team's score minus the opponent's
  marginMax: number;
};

export type ZoneSplit = {
  shots: number;
  makes: number;
  points: number;
  points_per_shot: number | null;
  small_sample: boolean;
};

export type ClutchRow = { zone: string; label: string; team: ZoneSplit; opp: ZoneSplit };

export type ClutchReport = {
  filters: { season: string; team: string; opp: string; period_from: number; seconds_left: number; margin_min: number; margin_max: number };
  small_sample_threshold: number;
  totals: { team: ZoneSplit; opp: ZoneSplit };
  rows: ClutchRow[];
  message: string | null;
  dataset_version: string | null;
};

export const MAX_PERIOD = 4;
export const MAX_SECONDS_LEFT = 12 * 60;
export const MAX_MARGIN = 40;

export const DEFAULT_CLUTCH: Omit<ClutchFilters, "season" | "team" | "opp"> = {
  periodFrom: 4,
  secondsLeft: 300,
  marginMin: -5,
  marginMax: 5,
};

/** Returns a message for the first filter that cannot be used, or "" when the report can run. */
export function clutchFilterProblem(f: ClutchFilters): string {
  if (!f.season.trim()) return "Pick a season.";
  if (!f.team.trim() || !f.opp.trim()) return "Pick your team and the opponent.";
  if (f.team.trim().toUpperCase() === f.opp.trim().toUpperCase()) return "Pick two different teams.";
  if (!Number.isInteger(f.periodFrom) || f.periodFrom < 1 || f.periodFrom > MAX_PERIOD)
    return `Start period must be between 1 and ${MAX_PERIOD} (overtime is always included).`;
  if (!Number.isInteger(f.secondsLeft) || f.secondsLeft < 1 || f.secondsLeft > MAX_SECONDS_LEFT)
    return "Time left must be between 1 second and 12 minutes.";
  for (const m of [f.marginMin, f.marginMax])
    if (!Number.isInteger(m) || Math.abs(m) > MAX_MARGIN) return `Score margin must be between -${MAX_MARGIN} and ${MAX_MARGIN}.`;
  if (f.marginMin > f.marginMax) return "The lowest margin must not be above the highest margin.";
  return "";
}

export function describeClutch(f: ClutchFilters): string {
  const periods = f.periodFrom === 4 ? "4th quarter and overtime" : `period ${f.periodFrom} onward, including overtime`;
  const sign = (n: number) => (n > 0 ? `+${n}` : String(n));
  return `${periods}, ${formatClock(f.secondsLeft)} or less left, margin ${sign(f.marginMin)} to ${sign(f.marginMax)}`;
}

/**
 * The Quick-Call situation for this report: the latest period in the range,
 * the time limit, and the margin in the range closest to a tie.
 */
export function clutchQuickCallSituation(f: ClutchFilters): QuickCallSituation {
  const margin = f.marginMin > 0 ? f.marginMin : f.marginMax < 0 ? f.marginMax : 0;
  return clampSituation({
    season: f.season,
    ourTeam: f.team.toUpperCase(),
    oppTeam: f.opp.toUpperCase(),
    margin,
    period: Math.max(4, f.periodFrom),
    timeRemaining: f.secondsLeft,
  });
}
