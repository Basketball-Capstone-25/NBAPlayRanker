// app/domain/weight-calculation.ts
//
// Domain Layer — Weight Calculation subsystem
// Ensures offensive + defensive weights sum to 1.

/** Scale the two weights to sum to one. A nonpositive or nonfinite total uses the default 50/50 split. */
export function normalizeWeights(wOff: number, wDef: number) {
  const a = Number(wOff);
  const b = Number(wDef);
  const sum = a + b;
  if (!Number.isFinite(sum) || sum <= 0) return { wOff: 0.5, wDef: 0.5 };
  return { wOff: a / sum, wDef: b / sum };
}
