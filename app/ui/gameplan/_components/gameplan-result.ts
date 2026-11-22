import type { ResolvedSituation } from "../../../domain/gameplan-context";

export type GameplanSelection = {
  season: string;
  our: string;
  opp: string;
  k: number;
  wOff: number;
  wDef: number;
  situation: ResolvedSituation | null;
  prompt: string;
};

export function gameplanSelectionKey(selection: GameplanSelection): string {
  const situation = selection.situation;
  // An omitted condition and an unchecked condition mean the same thing.
  const conditions = { ...situation?.coaching };
  if (conditions.need3 || conditions.need === "need3") {
    conditions.need3 = true;
    conditions.need = "need3";
  }
  if (conditions.vs_switching || conditions.defense_style === "switch") {
    conditions.vs_switching = true;
    conditions.defense_style = "switch";
  }
  const coaching = Object.entries(conditions)
    .filter(([, value]) => value !== false && value !== undefined)
    .sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify([
    selection.season, selection.our, selection.opp, selection.k, selection.wOff, selection.wDef,
    situation?.period ?? null, situation?.time_remaining ?? null, situation?.margin ?? null,
    situation?.end_of_quarter ?? false, coaching, selection.prompt.trim(),
  ]);
}

export function selectGameplanResult(
  completed: GameplanSelection | null,
  draft: GameplanSelection,
  state: { loading: boolean; parsing: boolean; shotLoading: boolean; shotError: string | null; hasShots: boolean },
) {
  const changed = completed !== null && gameplanSelectionKey(completed) !== gameplanSelectionKey(draft);
  const current = completed !== null && !changed && !state.loading && !state.parsing;
  return {
    changed,
    current,
    inputs: current ? completed : null,
    shotsCurrent: current && !state.shotLoading && !state.shotError && state.hasShots,
  };
}
