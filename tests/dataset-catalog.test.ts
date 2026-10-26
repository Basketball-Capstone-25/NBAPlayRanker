import { describe, expect, it } from "vitest";
import { parseDatasetCatalog } from "../app/infrastructure/dataset-catalog";
import { baselineReport, contextReport, metricsReport } from "../app/infrastructure/report-export";

describe("dataset metadata", () => {
  it("keeps an unknown pull date and valid source count", () => {
    const datasets = [{ id: "synergy", name: "Play types", version: `sha256:${"a".repeat(64)}`,
      pulled_at: null, row_count: 10, seasons: ["2024-25"], columns: ["PPP"] }];
    expect(parseDatasetCatalog({ datasets })[0].pulled_at).toBeNull();
    expect(parseDatasetCatalog({ datasets })[0].row_count).toBe(10);
  });
  it("rejects malformed metadata", () => {
    expect(() => parseDatasetCatalog({ datasets: [{ id: "x", name: "x", row_count: -1 }] })).toThrow("unavailable");
  });
  it("keeps the displayed response version in exports", () => {
    const versions = { synergy: `sha256:${"a".repeat(64)}` };
    const row = { playType: "Cut", pppPred: 1.1, pppOff: 1.1, pppDef: 1, pppGap: .1, raw: { dataset_versions: versions } };
    expect(baselineReport({}, [row]).metadata.dataset_versions).toEqual(versions);
    expect(metricsReport({ n_splits: 5, metrics: [], rf_vs_baseline_p: null, rf_vs_baseline_t: null,
      dataset_versions: versions }).metadata.dataset_versions).toEqual(versions);
    const snapshot = contextReport({}, [{ playType: "Cut", finalPPP_display: 1.2, mlPPP: 1.1,
      baselinePPP: 1, deltaPPP_display: .2, overlay: { overlayPPP: .1 }, raw: { dataset_versions: versions } }]);
    versions.synergy = `sha256:${"b".repeat(64)}`;
    expect(snapshot.metadata.dataset_versions).toEqual({ synergy: `sha256:${"a".repeat(64)}` });
  });
});
