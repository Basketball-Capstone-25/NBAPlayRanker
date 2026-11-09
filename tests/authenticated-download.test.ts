import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession } }) }));

import { API_BASE, authenticatedDownload } from "../app/infrastructure/api-client";
import { clutchFilename, clutchReportUrl } from "../app/infrastructure/clutch-report";

describe("authenticated HTTP downloads", () => {
  const anchor = { href: "", target: "", rel: "", click: vi.fn(), remove: vi.fn() };
  const createElement = vi.fn(() => anchor);
  const appendChild = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    getSession.mockResolvedValue({ data: { session: { access_token: "test-session-token" } } });
    vi.stubGlobal("document", { createElement, body: { appendChild } });
    vi.spyOn(URL, "createObjectURL");
  });
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it("opens a same-origin HTTP attachment without putting a token in the URL", async () => {
    const query = "season=2024-25&team=TOR&play_type=Cut%20%26%20Roll";
    await authenticatedDownload(`${API_BASE}/data/team-playtypes.csv?${query}`, "team_TOR.csv");

    const target = new URL(anchor.href, "https://frontend.example.test");
    expect(target.origin).toBe("https://frontend.example.test");
    expect(target.pathname).toBe("/api/exports");
    expect(target.searchParams.get("path")).toBe("/data/team-playtypes.csv");
    expect(target.searchParams.get("query")).toBe(query);
    expect(target.searchParams.get("filename")).toBe("team_TOR.csv");
    expect(anchor.href).not.toContain("test-session-token");
    expect(anchor).not.toHaveProperty("download");
    expect(anchor.target).toBe("_blank");
    expect(anchor.rel).toBe("noopener noreferrer");
    expect(appendChild).toHaveBeenCalledWith(anchor);
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(anchor.remove).toHaveBeenCalledOnce();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("rejects a missing session before navigating", async () => {
    getSession.mockResolvedValue({ data: { session: null } });
    await expect(authenticatedDownload(`${API_BASE}/rank-plays/baseline.csv`)).rejects.toThrow("Sign in");
    expect(createElement).not.toHaveBeenCalled();
  });

  it.each(["csv", "pdf"] as const)("opens the clutch %s through the authenticated proxy", async format => {
    const filters = { season: "2024-25", team: "TOR", opp: "BOS", periodFrom: 4,
      secondsLeft: 300, marginMin: -5, marginMax: 5 };
    await authenticatedDownload(clutchReportUrl(filters, format), clutchFilename(filters, format));
    const target = new URL(anchor.href, "https://frontend.example.test");
    expect(target.pathname).toBe("/api/exports");
    expect(target.searchParams.get("path")).toBe(`/data/clutch-report.${format}`);
    expect(target.searchParams.get("query")).toBe("season=2024-25&team=TOR&opp=BOS&period_from=4&seconds_left=300&margin_min=-5&margin_max=5");
    expect(target.searchParams.get("filename")).toBe(`clutch_TOR_vs_BOS_2024-25.${format}`);
    expect(anchor.href).not.toContain("test-session-token");
    expect(anchor.click).toHaveBeenCalledOnce();
  });

  it.each(["https://foreign.example.test/data/team-playtypes.csv", `${API_BASE}/meta/options`])(
    "rejects unsupported export targets: %s", async url => {
      await expect(authenticatedDownload(url)).rejects.toThrow("Unsupported export URL");
      expect(createElement).not.toHaveBeenCalled();
    }
  );

  it("removes the temporary link if browser navigation fails", async () => {
    anchor.click.mockImplementationOnce(() => { throw new Error("Download blocked"); });
    await expect(authenticatedDownload(`${API_BASE}/export/shotplan.pdf`)).rejects.toThrow("Download blocked");
    expect(anchor.remove).toHaveBeenCalledOnce();
  });
});
