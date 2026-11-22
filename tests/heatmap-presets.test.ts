import { describe, expect, it } from "vitest";
import { heatmapPresetFilters } from "../app/ui/shot-analysis/_components/heatmapPresets";

const zones = ["arc3", "corner3", "mid", "paint", "rim", "unknown"];
describe("heatmap zone presets", () => {
  it.each([["RIM", "rim"], ["CORNER_3", "corner3"], ["ABOVE_BREAK_3", "arc3"], ["MIDRANGE", "mid"]] as const)("maps %s to its exact court region", (preset, zone) => {
    expect(heatmapPresetFilters(preset, zones)).toEqual({ shotType: "", zone });
  });
  it("does not call corner threes above-the-break shots when arc data is missing", () => {
    expect(heatmapPresetFilters("ABOVE_BREAK_3", ["corner3", "mid"])).toBeNull();
  });
  it("does not replace rim attempts with all paint attempts", () => {
    expect(heatmapPresetFilters("RIM", ["paint"])).toBeNull();
  });
  it("keeps a missing requested region unavailable instead of clearing to All", () => {
    expect(heatmapPresetFilters("CORNER_3", [])).toBeNull();
    expect(heatmapPresetFilters("ALL", [])).toEqual({ shotType: "", zone: "" });
  });
  it("preserves a recognized source label without a substring match", () => {
    expect(heatmapPresetFilters("ABOVE_BREAK_3", ["Above the Break 3"])).toEqual({ shotType: "", zone: "Above the Break 3" });
    expect(heatmapPresetFilters("RIM", ["Restricted Area"])).toEqual({ shotType: "", zone: "Restricted Area" });
  });
});
