import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

import {
  canSearchShots,
  fetchShotPage,
  ShotAccessError,
  ShotDataUnavailableError,
  shotExplorerUrl,
  shotExportFilename,
  shotExportUrl,
} from "../app/infrastructure/shot-explorer";
import { EXPORT_PATHS, exportFilename, localExportUrl } from "../app/infrastructure/export-contract";

const filters = { season: "2024-25", team: "TOR", opp: "BOS", shotType: "", zone: " ", result: "made" as const };

describe("Data Explorer play-by-play shots client (SCRUM-627)", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", fetchMock);
    getSession.mockResolvedValue({ data: { session: { access_token: "analyst-token" } } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("needs a season and team before searching", () => {
    expect(canSearchShots({ season: "2024-25", team: "" })).toBe(false);
    expect(canSearchShots({ season: " ", team: "TOR" })).toBe(false);
    expect(canSearchShots(filters)).toBe(true);
  });

  it("sends only the filters that are set, plus 50-row paging", () => {
    const url = new URL(shotExplorerUrl(filters, 3));
    expect(url.pathname).toMatch(/\/data\/pbp-shots$/);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      season: "2024-25", team: "TOR", opp: "BOS", result: "made", page: "3", page_size: "50",
    });
  });

  it("exports the same filters as CSV or JSON through the allowed export proxy", () => {
    const csv = shotExportUrl(filters, "csv");
    const json = shotExportUrl(filters, "json");
    expect(new URL(csv).pathname).toMatch(/\/data\/pbp-shots\.csv$/);
    expect(new URL(json).searchParams.get("opp")).toBe("BOS");
    expect(EXPORT_PATHS.has("/data/pbp-shots.csv") && EXPORT_PATHS.has("/data/pbp-shots.json")).toBe(true);
    const base = new URL(csv).origin;
    expect(localExportUrl(json, base, shotExportFilename(filters, "json"))).toContain("filename=pbp_shots_2024-25_TOR_BOS.json");
    expect(exportFilename("/data/pbp-shots.json", "shots")).toBe("shots.json");
  });

  it("sends the analyst's token and returns the page", async () => {
    const page = { columns: [], rows: [], page: 1, page_size: 50, total: 0, has_next: false, dataset_version: null };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(page), { status: 200 }));
    await expect(fetchShotPage(filters, 1)).resolves.toEqual(page);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer analyst-token" });
  });

  it("maps 403, 503 and 400 to clear messages", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Role 'coach' cannot access" }), { status: 403 }));
    await expect(fetchShotPage(filters, 1)).rejects.toBeInstanceOf(ShotAccessError);
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Shot data is not available yet." }), { status: 503 }));
    await expect(fetchShotPage(filters, 1)).rejects.toBeInstanceOf(ShotDataUnavailableError);
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Result must be made or missed." }), { status: 400 }));
    await expect(fetchShotPage(filters, 1)).rejects.toThrow("Result must be made or missed.");
  });
});
