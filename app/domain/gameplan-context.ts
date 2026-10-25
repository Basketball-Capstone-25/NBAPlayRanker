export const NEEDS = ["need3", "quick2", "two_for_one", "last_shot", "must_score", "safe"] as const;
export const DEFENSE_STYLES = ["switch", "drop", "hedge", "blitz", "ice", "under", "top_lock", "zone_2_3", "zone_3_2", "zone_1_3_1", "box_and_1", "matchup_zone", "generic_zone"] as const;
export const CONDITION_FLAGS = ["after_timeout", "late_clock", "need3", "protect_lead", "vs_switching", "must_stop", "quick2", "two_for_one", "slob", "blob", "advance_ball", "hold_for_last", "safe", "must_score", "no_three"] as const;

export type CoachingConditions = Partial<Record<typeof CONDITION_FLAGS[number], boolean>> & {
  need?: typeof NEEDS[number];
  defense_style?: typeof DEFENSE_STYLES[number];
  pace?: "push" | "slow";
  shot_clock?: number;
};
export type GameSituation = { period: number; time_remaining: number; margin: number };
export type ParsedSituation = Record<string, unknown>;
export type ResolvedSituation = GameSituation & {
  coaching: CoachingConditions;
  end_of_quarter: boolean;
  warnings: string[];
};

export function periodLabel(period: number): string {
  return period === 5 ? "OT" : `Q${period}`;
}

export function periodSeconds(period: number): number {
  return period === 5 ? 300 : 720;
}

export function resolveGameSituation(parsed: ParsedSituation, fallback: GameSituation): ResolvedSituation {
  const finite = (value: unknown, previous: number) => typeof value === "number" && Number.isFinite(value) ? value : previous;
  const period = Math.max(1, Math.min(5, Math.round(finite(parsed.period, fallback.period))));
  const rawClock = finite(parsed.time_remaining, fallback.time_remaining);
  const time_remaining = Math.max(0, Math.min(periodSeconds(period), rawClock));
  const warnings: string[] = [];
  if (rawClock !== time_remaining) warnings.push(`Clock adjusted to ${time_remaining} seconds for ${periodLabel(period)}.`);
  const coaching: CoachingConditions = {};
  for (const key of CONDITION_FLAGS) {
    if (typeof parsed[key] === "boolean") coaching[key] = parsed[key];
  }
  if (typeof parsed.need === "string" && NEEDS.includes(parsed.need as typeof NEEDS[number])) {
    coaching.need = parsed.need as typeof NEEDS[number];
  } else if (parsed.need && !["stop", "no_three"].includes(String(parsed.need))) {
    warnings.push(`“${String(parsed.need).replaceAll("_", " ")}” is a note only; it does not change the ranking.`);
  }
  if (typeof parsed.defense_style === "string") {
    if (DEFENSE_STYLES.includes(parsed.defense_style as typeof DEFENSE_STYLES[number])) coaching.defense_style = parsed.defense_style as typeof DEFENSE_STYLES[number];
    else warnings.push(`“${parsed.defense_style.replaceAll("_", " ")}” coverage is not supported and does not change the ranking.`);
  }
  if (parsed.pace === "push" || parsed.pace === "slow") coaching.pace = parsed.pace;
  else if (parsed.pace) warnings.push(`“${String(parsed.pace)}” pace is not supported and does not change the ranking.`);
  if (typeof parsed.shot_clock === "number" && Number.isFinite(parsed.shot_clock)) coaching.shot_clock = Math.max(0, Math.min(24, parsed.shot_clock));
  if (parsed.end_of_quarter) warnings.push("End of period is a note. The ranking uses the period and remaining seconds, with no separate bonus for this label.");
  if (parsed.foul_game) warnings.push("Foul-game instructions are a note only; they do not change the ranking.");
  return {
    period, time_remaining,
    margin: Math.max(-50, Math.min(50, finite(parsed.margin, fallback.margin))),
    coaching, end_of_quarter: Boolean(parsed.end_of_quarter), warnings,
  };
}

// The returned snapshot is used by this Build, without waiting for React state.
export async function resolveBuildSituation(
  text: string,
  current: ResolvedSituation,
  parse: (text: string, defaults: GameSituation) => Promise<ParsedSituation>,
): Promise<ResolvedSituation> {
  if (!text.trim()) return current;
  return resolveGameSituation(await parse(text, current), current);
}
