import { describe, expect, it } from "vitest";
import type { TeamComparison } from "../app/domain/team-comparison";
import { profileDifferenceText, profileValueText } from "../app/domain/profile-display";

const data = { team_possessions: 1_000_000, opponent_possessions: 1_000_000 } as TeamComparison;
const row = (points: number): TeamComparison["rows"][number] => ({ play_type: "Cut",
  team: { points, possessions: 1_000_000, ppp: points / 1_000_000, usage_pct: 100 },
  opponent: { points: 1_000_000, possessions: 1_000_000, ppp: 1, usage_pct: 100 },
  difference: { ppp: points / 1_000_000 - 1, usage_pp: 0 } });

describe("profile display rounding from source counts", () => {
  it.each([[1_137_499, "13.7"], [1_137_500, "13.8"], [1_137_501, "13.8"],
    [862_499, "-13.8"], [862_500, "-13.8"], [862_501, "-13.7"]] as const)("rounds %s points without nudging rates near the tie", (points, expected) => {
    expect(profileDifferenceText(data, row(points), "ppp")).toBe(expected);
  });
  it("formats the scoring rate and usage from their own denominators", () => {
    const sample = row(1_137_500);
    expect(profileValueText(data, sample, "team", "ppp")).toBe("113.8");
    expect(profileValueText(data, sample, "team", "ppp", 3, false)).toBe("1.138");
    const totals = { ...data, team_possessions: 8_000_000 };
    expect(profileValueText(totals, sample, "team", "usage")).toBe("12.5");
    expect(profileDifferenceText(totals, sample, "usage")).toBe("-87.5");
  });
  it("keeps missing and zero denominators unavailable and suppresses negative zero", () => {
    expect(profileDifferenceText(data, row(999_999), "ppp")).toBe("0.0");
    expect(profileDifferenceText(data, { ...row(1), opponent: null }, "ppp")).toBeNull();
    const empty = row(0); empty.team = { points: 0, possessions: 0, ppp: null, usage_pct: 0 };
    expect(profileValueText(data, empty, "team", "ppp")).toBeNull();
    expect(profileValueText(data, empty, "team", "usage")).toBe("0.0");
  });
});
