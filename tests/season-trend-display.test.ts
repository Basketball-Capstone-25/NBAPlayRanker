import { describe, expect, it } from "vitest";
import { trendAxis, trendChange } from "../app/domain/trend-display";
import type { TrendRow } from "../app/domain/season-trend";

const row = (season: string, ppp: number | null, usage_pct: number | null = 10, possessions = 100): TrendRow => ({
  season, ppp, usage_pct, possessions: ppp === null ? null : possessions,
  points: ppp === null ? null : ppp * possessions, team_possessions: 1000,
});

describe("season trend scale", () => {
  it("pads a constant series and gives tiny changes a minimum context", () => {
    const constant = trendAxis([row("2023-24", 1.2), row("2024-25", 1.2)], "ppp", "focused");
    expect(constant.minimum).toBeCloseTo(1.15);
    expect(constant.maximum).toBeCloseTo(1.25);
    const tiny = trendAxis([row("2023-24", 1.2), row("2024-25", 1.20001)], "ppp", "focused");
    expect(tiny.maximum - tiny.minimum).toBeCloseTo(0.1);
  });
  it("keeps usage inside 0–100 and all observed values inside the range", () => {
    for (const values of [[0, 0], [99.9, 100], [0, 100], [10, 11]]) {
      const rows = values.map((value, index) => row(`${2023 + index}-24`, 1, value));
      const axis = trendAxis(rows, "usage_pct", "focused");
      expect(axis.minimum).toBeGreaterThanOrEqual(0);
      expect(axis.maximum).toBeLessThanOrEqual(100);
      expect(axis.maximum).toBeGreaterThan(axis.minimum);
      expect(axis.minimum).toBeLessThanOrEqual(Math.min(...values));
      expect(axis.maximum).toBeGreaterThanOrEqual(Math.max(...values));
    }
  });
  it("offers a true zero baseline without changing the input", () => {
    const rows = [row("2023-24", 1.12), row("2024-25", 1.18)];
    const before = structuredClone(rows);
    expect(trendAxis(rows, "ppp", "zero")).toEqual({ minimum: 0, maximum: 1.25 });
    expect(rows).toEqual(before);
  });
  it("uses a finite fallback for all missing data and contains unusual high PPP", () => {
    expect(trendAxis([row("2024-25", null, null)], "ppp", "focused")).toEqual({ minimum: 0, maximum: 1 });
    const axis = trendAxis([row("2024-25", 4)], "ppp", "focused");
    expect(axis.minimum).toBeLessThan(4); expect(axis.maximum).toBeGreaterThan(4);
  });
});

describe("season trend explanation", () => {
  it("converts PPP change into points per 100 of that play type", () => {
    const change = trendChange([row("2023-24", 1.12), row("2024-25", 1.17)], "ppp", 20)!;
    expect(change.displayedDifference).toBe(5);
    expect(change.direction).toBe("up");
    expect(change.difference).toBeCloseTo(0.05);
  });
  it("uses percentage points for usage rather than relative growth", () => {
    const change = trendChange([row("2023-24", 1, 10), row("2024-25", 1, 12)], "usage_pct", 20)!;
    expect(change.displayedDifference).toBe(2); expect(change.direction).toBe("up");
  });
  it("does not assign a direction to differences hidden by display precision", () => {
    for (const delta of [-0.0001, 0, 0.0001]) {
      const change = trendChange([row("2023-24", 1.2), row("2024-25", 1.2 + delta)], "ppp", 20)!;
      expect(change.direction).toBe("same"); expect(change.displayedDifference).toBe(0);
    }
  });
  it("keeps first/last recorded seasons, warns about gaps and gates small samples", () => {
    const rows = [row("2021-22", null, null), row("2022-23", 1.2, 10, 19),
      row("2023-24", null, null), row("2024-25", 1.1), row("2025-26", null, null)];
    const change = trendChange(rows, "ppp", 20)!;
    expect(change.first.season).toBe("2022-23"); expect(change.last.season).toBe("2024-25");
    expect(change.missingBetween).toBe(true); expect(change.smallSample).toBe(true);
    expect(trendChange(rows, "ppp", 19)!.smallSample).toBe(false);
  });
  it("does not invent a comparison for one or zero recorded seasons", () => {
    expect(trendChange([row("2024-25", 1)], "ppp", 20)).toBeNull();
    expect(trendChange([row("2024-25", null, null)], "ppp", 20)).toBeNull();
  });
});
