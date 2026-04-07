// app/domain/weight-calculation.ts
//
// Domain Layer — Weight Calculation subsystem
// Ensures offensive + defensive weights sum to 1.

export function normalizeWeights(wOff: number, wDef: number) {
  const a = Number(wOff);
  const b = Number(wDef);
  const sum = a + b;
  if (!Number.isFinite(sum) || sum <= 0) return { wOff: 0.7, wDef: 0.3 };
  return { wOff: a / sum, wDef: b / sum };
}
