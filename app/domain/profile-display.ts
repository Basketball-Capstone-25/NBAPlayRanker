import type { TeamComparison } from "./team-comparison";

type Row = TeamComparison["rows"][number];
type Metric = "ppp" | "usage";
type Fraction = { numerator: bigint; denominator: bigint };

function fraction(data: TeamComparison, row: Row, side: "team" | "opponent", metric: Metric, per100: boolean): Fraction | null {
  const sample = row[side];
  if (!sample) return null;
  const denominator = metric === "ppp" ? sample.possessions : data[`${side}_possessions`];
  if (!denominator) return null;
  const numerator = metric === "ppp" ? sample.points : sample.possessions;
  return { numerator: BigInt(numerator) * BigInt(metric === "usage" || per100 ? 100 : 1), denominator: BigInt(denominator) };
}

/** Round the integer source ratio once, with half ties away from zero. */
function fixed(value: Fraction, digits: number): string {
  const scale = 10n ** BigInt(digits), negative = value.numerator < 0n;
  const magnitude = negative ? -value.numerator : value.numerator;
  const rounded = (2n * magnitude * scale + value.denominator) / (2n * value.denominator);
  return `${negative && rounded !== 0n ? "-" : ""}${rounded / scale}.${String(rounded % scale).padStart(digits, "0")}`;
}

export function profileValueText(data: TeamComparison, row: Row, side: "team" | "opponent", metric: Metric, digits = 1, per100 = true): string | null {
  const value = fraction(data, row, side, metric, per100);
  return value ? fixed(value, digits) : null;
}

export function profileDifferenceText(data: TeamComparison, row: Row, metric: Metric, digits = 1, per100 = true): string | null {
  const a = fraction(data, row, "team", metric, per100), b = fraction(data, row, "opponent", metric, per100);
  return a && b ? fixed({ numerator: a.numerator * b.denominator - b.numerator * a.denominator,
    denominator: a.denominator * b.denominator }, digits) : null;
}
