import { afterEach, describe, expect, it, vi } from "vitest";
import { formatBaselineNumber } from "../app/domain/baseline-metrics";
import { normalizePlay } from "../app/services/gameplan";
import { baselineRank } from "../app/infrastructure/api-client";
import { baselineReport } from "../app/infrastructure/report-export";
import { topPickIndexes } from "../app/domain/top-pick";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));
afterEach(() => vi.unstubAllGlobals());

async function rankings(scores: unknown[]) {
  const rows = scores.map((score, i) => ({
    PLAY_TYPE: `Play ${i}`, PPP_PRED: score, PPP_OFF_SHRUNK: null, PPP_DEF_SHRUNK: "0", PPP_GAP: " ",
  }));
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ rankings: rows })));
  return baselineRank({ season: "2024-25", our: "TOR", opp: "BOS" });
}

describe("Baseline with missing rates", () => {
  it("retains unavailable values in the rows and exports real zeros as numbers", async () => {
    const [row] = await rankings([null]);
    expect([row.pppPred, row.pppOff, row.pppDef, row.pppGap]).toEqual([null, null, 0, null]);
    expect(formatBaselineNumber(row.pppPred)).toBe("Unavailable");
    expect(formatBaselineNumber(row.pppDef)).toBe("0.000");
    expect(normalizePlay(row, 0, "baseline")).toMatchObject({ ppp: null, defense: 0 });
    expect(baselineReport({}, [row]).rows[0]).toMatchObject({
      "Predicted PPP": null, "Our PPP": null, "Opponent PPP": 0, "PPP gap": null,
    });
  });

  it("formats numeric strings, missing values and invalid inputs consistently", () => {
    for (const value of [null, undefined, "", " ", false, Number.NaN, Infinity]) {
      expect(formatBaselineNumber(value)).toBe("Unavailable");
    }
    expect(formatBaselineNumber("0", 0)).toBe("0");
    expect(formatBaselineNumber("-0.045")).toBe("-0.045");
  });

  it("keeps rows with missing scores after the ranked plays without creating a false tie at zero", async () => {
    const rows = await rankings([null, "0", undefined, 0, "bad"]);
    expect(rows.map(r => r.playType)).toEqual(["Play 1", "Play 3", "Play 0", "Play 2", "Play 4"]);
    expect(topPickIndexes(rows.map(r => r.pppPred))).toEqual([0, 1]);
  });
});
