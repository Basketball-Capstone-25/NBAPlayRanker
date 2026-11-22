import { describe, expect, it } from "vitest";
import { rmseBarWidths } from "../app/ui/analytics/_components/rmse-bars";

function widths(values: number[]) {
  return rmseBarWidths(values.map((RMSE_mean, index) => ({ model: String(index), RMSE_mean })))
    .map((bar) => bar.widthPct);
}

describe("holdout RMSE bars", () => {
  it("makes lower error shorter on a shared zero-based scale", () => {
    const result = widths([0.06, 0.08, 0.1]);
    expect(result[0]).toBeCloseTo(60);
    expect(result[1]).toBeCloseTo(80);
    expect(result[2]).toBe(100);
  });

  it("draws a zero error with zero width, including an all-zero run", () => {
    expect(widths([0, 0.1])).toEqual([0, 100]);
    expect(widths([0, 0])).toEqual([0, 0]);
  });

  it("gives equal positive errors equal widths without dividing by their difference", () => {
    expect(widths([0.07, 0.07])).toEqual([100, 100]);
  });

  it("marks invalid errors unavailable without changing the valid scale", () => {
    expect(widths([NaN, Infinity, -Infinity, -1, 0.1, 0.2]))
      .toEqual([null, null, null, null, 50, 100]);
    expect(widths([NaN, -1])).toEqual([null, null]);
  });

  it("keeps model order and leaves the measured values unchanged", () => {
    const rows = Object.freeze([
      Object.freeze({ model: "Baseline", RMSE_mean: 0.12 }),
      Object.freeze({ model: "Ridge", RMSE_mean: 0.06 }),
    ]);
    expect(rmseBarWidths(rows)).toEqual([
      { model: "Baseline", widthPct: 100 },
      { model: "Ridge", widthPct: 50 },
    ]);
    expect(rows.map((row) => row.RMSE_mean)).toEqual([0.12, 0.06]);
  });

  it("has no bars without results", () => {
    expect(widths([])).toEqual([]);
  });
});
