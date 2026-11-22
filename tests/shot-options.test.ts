import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ shots: vi.fn(), labels: vi.fn() }));
vi.mock("../app/services/shotAnalysis", () => ({ fetchPbpMetaOptions: mocks.shots, fetchMetaOptions: mocks.labels }));
import { loadShotOptions } from "../app/ui/shot-analysis/_components/shotOptions";

const shots = { seasons: ["2024-25"], teams: ["TOR", "BOS"], shotTypes: ["Dunk"], zones: ["rim"] };
beforeEach(() => {
  vi.resetAllMocks();
  mocks.shots.mockResolvedValue(shots);
  mocks.labels.mockResolvedValue({ seasons: ["2019-20"], teams: ["LAL"], teamNames: { TOR: "Toronto Raptors" } });
});

describe("shot-source dropdown options", () => {
  it("uses only shot-source choices and borrows names without merging play-type options", async () => {
    expect(await loadShotOptions()).toEqual({ ...shots, teamNames: { TOR: "Toronto Raptors" } });
    expect(mocks.shots).toHaveBeenCalledWith({ strict: true });
  });
  it("does not substitute another dataset when shot metadata fails", async () => {
    mocks.shots.mockRejectedValue(new Error("Shot source unavailable"));
    await expect(loadShotOptions()).rejects.toThrow("Shot source unavailable");
  });
  it("keeps valid shot choices when optional descriptive team names are unavailable", async () => {
    mocks.labels.mockRejectedValue(new Error("Play-type source unavailable"));
    expect(await loadShotOptions()).toEqual({ ...shots, teamNames: {} });
  });
  it.each(["seasons", "teams"])("reports an empty %s source without inventing choices", async key => {
    mocks.shots.mockResolvedValue({ ...shots, [key]: [] });
    await expect(loadShotOptions()).rejects.toThrow("No shot seasons or teams are available");
  });
  it("keeps missing optional shot type and zone lists empty", async () => {
    mocks.shots.mockResolvedValue({ seasons: shots.seasons, teams: shots.teams });
    expect(await loadShotOptions()).toMatchObject({ shotTypes: [], zones: [] });
  });
});
