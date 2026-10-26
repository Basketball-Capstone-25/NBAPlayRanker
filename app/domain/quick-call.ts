// app/domain/quick-call.ts
//
// Domain Layer — Timeout Quick-Call Mode (SCRUM-628)
// The game situation a coach edits during a timeout, the +/- rules that keep
// it on a real scoreboard, and the preloaded late-game scenarios.

export type QuickCallSituation = {
  season: string;
  ourTeam: string;
  oppTeam: string;
  margin: number; // our score minus theirs
  period: number; // 1-4, 5 = overtime
  timeRemaining: number; // seconds left in the period
};

export type SituationField = "margin" | "period" | "timeRemaining";

export type LateGameScenario = {
  id: string;
  label: string;
  margin: number;
  period: number;
  timeRemaining: number;
};

export const MARGIN_LIMIT = 40;
export const OVERTIME_PERIOD = 5;
export const REGULATION_SECONDS = 720;
export const OVERTIME_SECONDS = 300;
export const TIME_STEP_SECONDS = 5;

export const LATE_GAME_SCENARIOS: LateGameScenario[] = [
  { id: "down3-under30", label: "Down 3, under 30s", margin: -3, period: 4, timeRemaining: 25 },
  { id: "down2-last-shot", label: "Down 2, last shot", margin: -2, period: 4, timeRemaining: 8 },
  { id: "down1-under60", label: "Down 1, under 1 min", margin: -1, period: 4, timeRemaining: 55 },
  { id: "tied-final", label: "Tied, final possession", margin: 0, period: 4, timeRemaining: 20 },
  { id: "up2-close-out", label: "Up 2, 30s left", margin: 2, period: 4, timeRemaining: 30 },
  { id: "down3-ot", label: "Down 3 in OT, 20s", margin: -3, period: 5, timeRemaining: 20 },
];

const clampTo = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function periodSeconds(period: number): number {
  return period >= OVERTIME_PERIOD ? OVERTIME_SECONDS : REGULATION_SECONDS;
}

/** Keep every value to something a scoreboard can show. */
export function clampSituation(s: QuickCallSituation): QuickCallSituation {
  const period = clampTo(Math.round(Number(s.period) || 4), 1, OVERTIME_PERIOD);
  return {
    ...s,
    margin: clampTo(Math.round(Number(s.margin) || 0), -MARGIN_LIMIT, MARGIN_LIMIT),
    period,
    timeRemaining: clampTo(Math.round(Number(s.timeRemaining) || 0), 0, periodSeconds(period)),
  };
}

/** One tap of a +/- button. Time moves in 5-second steps. */
export function stepSituation(s: QuickCallSituation, field: SituationField, direction: 1 | -1): QuickCallSituation {
  const delta = field === "timeRemaining" ? TIME_STEP_SECONDS * direction : direction;
  return clampSituation({ ...s, [field]: s[field] + delta });
}

/** Whether a +/- button would change anything (used to disable it at a limit). */
export function canStep(s: QuickCallSituation, field: SituationField, direction: 1 | -1): boolean {
  return stepSituation(s, field, direction)[field] !== s[field];
}

export function applyScenario(s: QuickCallSituation, scenario: LateGameScenario): QuickCallSituation {
  return clampSituation({ ...s, margin: scenario.margin, period: scenario.period, timeRemaining: scenario.timeRemaining });
}

/** Read the situation the Context ML page passed in the link. */
export function situationFromParams(params: URLSearchParams): QuickCallSituation {
  return clampSituation({
    season: params.get("season") ?? "",
    ourTeam: params.get("our") ?? "",
    oppTeam: params.get("opp") ?? "",
    margin: Number(params.get("margin") ?? 0),
    period: Number(params.get("period") ?? 4),
    timeRemaining: Number(params.get("time") ?? 60),
  });
}

export function quickCallHref(s: QuickCallSituation): string {
  const query = new URLSearchParams({
    season: s.season,
    our: s.ourTeam,
    opp: s.oppTeam,
    margin: String(s.margin),
    period: String(s.period),
    time: String(s.timeRemaining),
  });
  return `/context/quick-call?${query}`;
}

export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function formatPeriod(period: number): string {
  return period >= OVERTIME_PERIOD ? "OT" : `Q${period}`;
}

export function formatMargin(margin: number): string {
  if (margin === 0) return "Tied";
  return margin > 0 ? `Up ${margin}` : `Down ${Math.abs(margin)}`;
}
