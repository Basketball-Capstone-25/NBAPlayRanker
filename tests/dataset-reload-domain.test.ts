import { describe, expect, it } from "vitest";
import { DatasetReloadError, parseDatasetReload } from "../app/domain/dataset-reload";

const report = () => ({
  dataset: "synergy", status: "reloaded", source_file: "synergy.csv",
  previous_version: "version-one", dataset_version: "version-two", changed: true,
  rows: 47982, seasons: ["2023-24", "2024-25"], quality_status: "pass",
});

describe("dataset reload results", () => {
  it("accepts a changed source and copies its season list", () => {
    const source = report();
    const result = parseDatasetReload(source, "synergy");
    expect(result).toEqual(source);
    source.seasons.push("2025-26");
    expect(result.seasons).toEqual(["2023-24", "2024-25"]);
  });
  it("keeps an unchanged reload distinct from new data", () => {
    const source = { ...report(), previous_version: "version-two", changed: false };
    expect(parseDatasetReload(source, "synergy").changed).toBe(false);
  });
  it("accepts the shot contract without inventing a quality result", () => {
    const source = { ...report(), dataset: "pbp_shots", previous_version: "not loaded", source_file: "shots_clean.parquet" };
    delete (source as Partial<typeof source>).quality_status;
    expect(parseDatasetReload(source, "pbp")).toEqual(source);
    expect(parseDatasetReload({ ...source, quality_status: "pass" }, "pbp")).not.toHaveProperty("quality_status");
  });
  it.each([
    { dataset: "pbp_shots" }, { status: "running" }, { changed: false }, { changed: "true" },
    { previous_version: "" }, { dataset_version: "" }, { rows: 0 }, { rows: -1 }, { rows: 1.5 },
    { rows: Number.POSITIVE_INFINITY }, { rows: Number.MAX_SAFE_INTEGER + 1 }, { rows: "47982" },
    { seasons: [] }, { seasons: ["2024"] }, { seasons: ["2024-25", "2024-25"] }, { seasons: [null] },
    { source_file: "/private/server/data.csv" }, { source_file: "folder\\data.csv" }, { source_file: "" },
    { quality_status: "fail" }, { quality_status: undefined },
  ])("rejects inconsistent or missing fields: %j", change => {
    expect(() => parseDatasetReload({ ...report(), ...change }, "synergy")).toThrow(DatasetReloadError);
    try { parseDatasetReload({ ...report(), ...change }, "synergy"); } catch (error) {
      expect(error).toMatchObject({ outcome: "unknown" });
    }
  });
  it.each([null, undefined, [], "reloaded", 1])("rejects a non-report response: %j", value => {
    expect(() => parseDatasetReload(value, "synergy")).toThrow("incomplete");
  });
});
