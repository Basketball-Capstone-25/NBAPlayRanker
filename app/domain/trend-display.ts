import type { TrendRow } from "./season-trend";

export type TrendMetric = "ppp" | "usage_pct";
export type TrendScale = "focused" | "zero";

/** Keep a useful amount of context even when every rate is nearly identical. */
export function trendAxis(rows: TrendRow[], metric: TrendMetric, scale: TrendScale) {
  const values = rows.map(row => row[metric]).filter((value): value is number => value !== null && Number.isFinite(value));
  if (!values.length) return { minimum: 0, maximum: metric === "ppp" ? 1 : 100 };
  const low = Math.min(...values), high = Math.max(...values);
  if (scale === "zero") {
    const step = metric === "ppp" ? 0.25 : 5;
    return { minimum: 0, maximum: Math.max(metric === "ppp" ? 1 : 5, Math.ceil(high / step) * step) };
  }
  const minimumSpan = metric === "ppp" ? 0.1 : 5;
  const span = Math.max(minimumSpan, (high - low) * 1.4);
  let minimum = Math.max(0, (low + high - span) / 2);
  let maximum = minimum + span;
  if (metric === "usage_pct" && maximum > 100) {
    maximum = 100;
    minimum = Math.max(0, maximum - span);
  }
  return { minimum, maximum };
}

/** Compare the endpoints that actually have a value; never fill a missing season. */
export function trendChange(rows: TrendRow[], metric: TrendMetric, minimum: number) {
  const recorded = rows.filter(row => row[metric] !== null);
  if (recorded.length < 2) return null;
  const first = recorded[0], last = recorded[recorded.length - 1];
  const difference = last[metric]! - first[metric]!;
  // PPP × 100 is points per 100 possessions of this play type, not team PPG.
  const roundedDifference = Math.sign(difference) * Math.round((Math.abs(difference) * (metric === "ppp" ? 100 : 1) + 1e-10) * 10) / 10;
  return {
    first, last, difference,
    displayedDifference: Math.abs(roundedDifference),
    direction: roundedDifference === 0 ? "same" : roundedDifference > 0 ? "up" : "down",
    smallSample: (first.possessions ?? 0) < minimum || (last.possessions ?? 0) < minimum,
    missingBetween: rows.some(row => row.season > first.season && row.season < last.season && row[metric] === null),
  };
}
