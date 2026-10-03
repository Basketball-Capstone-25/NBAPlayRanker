import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSession } = vi.hoisted(() => ({ getSession: vi.fn() }));

vi.mock("../lib/supabase/client", () => ({
  createClient: () => ({ auth: { getSession } }),
}));

import { authenticatedDownload } from "../app/infrastructure/api-client";

describe("authenticated file downloads", () => {
  const fetchMock = vi.fn();
  const anchor = { href: "", download: "", click: vi.fn(), remove: vi.fn() };
  const createElement = vi.fn(() => anchor);
  const appendChild = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    anchor.href = "";
    anchor.download = "";
    getSession.mockResolvedValue({ data: { session: { access_token: "test-session-token" } } });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("document", { createElement, body: { appendChild } });
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test-download");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("sends the current session token and downloads CSV with the requested filename", async () => {
    const csv = "PLAY_TYPE,PPP\nCut,1.2\n";
    fetchMock.mockResolvedValue(new Response(csv, { headers: { "Content-Type": "text/csv" } }));

    await authenticatedDownload("https://api.example.test/data/team-playtypes.csv", "team_playtypes_2024-25_TOR.csv");

    expect(fetchMock).toHaveBeenCalledWith("https://api.example.test/data/team-playtypes.csv", {
      cache: "no-store",
      headers: { Authorization: "Bearer test-session-token" },
    });
    const downloadedBlob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
    expect(await downloadedBlob.text()).toBe(csv);
    expect(anchor.href).toBe("blob:test-download");
    expect(anchor.download).toBe("team_playtypes_2024-25_TOR.csv");
    expect(appendChild).toHaveBeenCalledWith(anchor);
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(anchor.remove).not.toHaveBeenCalled();
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(anchor.remove).toHaveBeenCalledOnce();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-download");
  });

  it("preserves the server filename for existing PDF download callers", async () => {
    fetchMock.mockResolvedValue(new Response("%PDF-fixture", {
      headers: { "Content-Type": "application/pdf", "Content-Disposition": 'attachment; filename="shot_plan.pdf"' },
    }));

    await authenticatedDownload("https://api.example.test/shot-plan.pdf");

    expect(anchor.download).toBe("shot_plan.pdf");
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer test-session-token");
  });

  it("keeps the URL's CSV filename when CORS hides Content-Disposition", async () => {
    fetchMock.mockResolvedValue(new Response("PLAY_TYPE,PPP\nCut,1.2\n"));

    await authenticatedDownload("https://api.example.test/rank/baseline.csv?season=2024-25&team=TOR");

    expect(anchor.download).toBe("baseline.csv");
    expect(anchor.click).toHaveBeenCalledOnce();
  });

  it("releases the blob even if the browser cannot trigger the download", async () => {
    fetchMock.mockResolvedValue(new Response("PLAY_TYPE,PPP\nCut,1.2\n"));
    anchor.click.mockImplementationOnce(() => { throw new Error("Download blocked"); });

    await expect(authenticatedDownload("https://api.example.test/export.csv"))
      .rejects.toThrow("Download blocked");

    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(anchor.remove).toHaveBeenCalledOnce();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-download");
  });

  it("surfaces a forbidden response without creating or downloading a file", async () => {
    const response = new Response('{"detail":"Analyst role required"}', { status: 403 });
    const readBlob = vi.spyOn(response, "blob");
    fetchMock.mockResolvedValue(response);

    await expect(authenticatedDownload("https://api.example.test/protected.csv", "export.csv"))
      .rejects.toThrow("Download failed (403)");

    expect(readBlob).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(createElement).not.toHaveBeenCalled();
    expect(anchor.click).not.toHaveBeenCalled();
  });
});
