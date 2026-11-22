import { describe, expect, it } from "vitest";
import { possessionScenario } from "../app/domain/possession-scenario";
import type { UsageEfficiencyRow } from "../app/domain/playtype-dashboard";

const row = (play_type: string, ppp: number): UsageEfficiencyRow => ({ play_type, ppp, possessions: 100, points: ppp * 100,
  usage_pct: 10, league_usage_pct: 10, league_ppp: 1, ppp_delta: ppp - 1, usage_delta_pp: 0, sample_ok: true });
describe("fixed-rate possession swap", () => {
  it("calculates points for a reallocation within 100 recorded possessions, not per game", () => {
    const result = possessionScenario(row("Isolation", .9), row("Cut", 1.2), 5)!;
    expect(result.pointsBefore).toBeCloseTo(4.5); expect(result.pointsAfter).toBeCloseTo(6);
    expect(result.difference).toBeCloseTo(1.5);
  });
  it("keeps negative, zero and unrounded differences", () => {
    expect(possessionScenario(row("A", 1.2), row("B", .9), 5)?.difference).toBeCloseTo(-1.5);
    expect(possessionScenario(row("A", 1), row("B", 1), 5)?.difference).toBe(0);
    expect(possessionScenario(row("A", 1.0004), row("B", 1.0006), 5)?.difference).toBeCloseTo(.001);
  });
  it.each([0, -1, NaN, Infinity, 11])("rejects an impossible amount %s", amount => {
    expect(possessionScenario(row("A", 1), row("B", 2), amount)).toBeNull();
  });
  it("requires distinct supported plays and respects actual source share", () => {
    expect(possessionScenario(row("A", 1), row("A", 2), 1)).toBeNull();
    expect(possessionScenario({ ...row("A", 1), sample_ok: false }, row("B", 2), 1)).toBeNull();
    expect(possessionScenario(row("A", 1), { ...row("B", 2), sample_ok: false }, 1)).toBeNull();
    expect(possessionScenario({ ...row("A", 1), usage_pct: .5 }, row("B", 2), 1)).toBeNull();
    expect(possessionScenario({ ...row("A", 1), usage_pct: .5 }, row("B", 2), .5)?.difference).toBe(.5);
  });
});
