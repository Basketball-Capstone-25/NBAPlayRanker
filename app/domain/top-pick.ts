// Domain Layer — which ranked plays count as the #1 pick (SCRUM-496 / UC24).
// Rows arrive already ranked. The first row is the top pick; any other row whose
// displayed score rounds to the same value is tied with it.

export const TOP_PICK_DECIMALS = 3;

function rounded(value: unknown, decimals: number): number | null {
  const n = Number(value);
  if (value === null || value === undefined || value === "" || !Number.isFinite(n)) return null;
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

/** Indexes of the top pick and every row tied with it on the displayed score. */
export function topPickIndexes(scores: ReadonlyArray<unknown>, decimals = TOP_PICK_DECIMALS): number[] {
  if (!scores.length) return [];
  const top = rounded(scores[0], decimals);
  if (top === null) return [0];
  return scores.flatMap((s, i) => (i === 0 || rounded(s, decimals) === top ? [i] : []));
}
