// app/domain/game-situation.ts
//
// Domain Layer — a game situation (score margin, period, time left) and the
// Context ML link that opens with it filled in (SCRUM-795).

export type GameSituation = {
  season: string;
  ourTeam: string;
  oppTeam: string;
  margin: number; // our score minus theirs
  period: number; // 1-4, 5 = overtime
  timeRemaining: number; // seconds left in the period
};

export const MARGIN_LIMIT = 40;
export const OVERTIME_PERIOD = 5;
export const REGULATION_SECONDS = 720;
export const OVERTIME_SECONDS = 300;

const clampTo = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function periodSeconds(period: number): number {
  return period >= OVERTIME_PERIOD ? OVERTIME_SECONDS : REGULATION_SECONDS;
}

/** Keep every value to something a scoreboard can show. */
export function clampSituation(s: GameSituation): GameSituation {
  const period = clampTo(Math.round(Number(s.period) || 4), 1, OVERTIME_PERIOD);
  return {
    ...s,
    margin: clampTo(Math.round(Number(s.margin) || 0), -MARGIN_LIMIT, MARGIN_LIMIT),
    period,
    timeRemaining: clampTo(Math.round(Number(s.timeRemaining) || 0), 0, periodSeconds(period)),
  };
}

/** Context ML with the matchup and situation already filled in. */
export function contextHref(s: GameSituation): string {
  const query = new URLSearchParams({
    season: s.season,
    our: s.ourTeam,
    opp: s.oppTeam,
    margin: String(s.margin),
    period: String(s.period),
    timeRemaining: String(s.timeRemaining),
  });
  return `/context?${query}`;
}

export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
