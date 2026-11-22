import { describe, expect, it, vi } from "vitest";
import { loadShotsForSeason } from "../app/ui/gameplan/_components/shot-coverage";

const seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"];
describe("Gameplan shot-source coverage", () => {
  it("does not call shot endpoints for the play-type-only 2019 season", async () => {
    const load = vi.fn().mockRejectedValue(new Error("404 then legacy422"));
    expect(await loadShotsForSeason("2019-20", seasons, load)).toBeNull();
    expect(load).not.toHaveBeenCalled();
  });
  it.each(seasons)("preserves the actual response for supported season %s", async (season) => {
    const data = { season, top_shot_types: [{ SHOT_TYPE: "Dunk", EPA: 1.7 }], top_zones: [] };
    const load = vi.fn().mockResolvedValue(data);
    expect(await loadShotsForSeason(season, seasons, load)).toBe(data);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("tries the requested season when metadata coverage is unknown", async () => {
    const load = vi.fn().mockResolvedValue({ top_shot_types: [], top_zones: [] });
    await loadShotsForSeason("2027-28", null, load);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("does not disguise a supported-season service error as missing data", async () => {
    await expect(loadShotsForSeason("2024-25", seasons, () => Promise.reject(new Error("unavailable")))).rejects.toThrow("unavailable");
  });
});
