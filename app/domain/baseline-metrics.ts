// Baseline responses may contain numeric strings or missing measurements.
export function baselineNumber(value: unknown): number | null {
  if (typeof value !== "number" && (typeof value !== "string" || value.trim() === "")) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/** Format a measured rate or count while keeping missing values distinct from zero. */
export function formatBaselineNumber(value: unknown, digits = 3): string {
  return baselineNumber(value)?.toFixed(digits) ?? "Unavailable";
}
