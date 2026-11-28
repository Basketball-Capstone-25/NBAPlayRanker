// Bar widths for the model error comparison.

type RmseRow = { model: string; RMSE_mean: number };

/** Scales errors from zero to the largest valid RMSE. Zero stays at zero, while an invalid error has no bar. */
export function rmseBarWidths(rows: readonly RmseRow[]) {
  const valid = (value: number) => Number.isFinite(value) && value >= 0;
  const maximum = Math.max(0, ...rows.map((row) => valid(row.RMSE_mean) ? row.RMSE_mean : 0));

  return rows.map((row) => ({
    model: row.model,
    widthPct: valid(row.RMSE_mean)
      ? maximum > 0 ? (row.RMSE_mean / maximum) * 100 : 0
      : null,
  }));
}
