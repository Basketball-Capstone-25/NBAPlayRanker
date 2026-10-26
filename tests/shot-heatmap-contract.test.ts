import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: async () => ({ data: { session: { access_token: "test-session" } } }) } }) }));
import { fetchShotHeatmap, getShotPlanPdfUrl } from "../app/infrastructure/api-client";

const filters = { season: "2024-25", team: "TOR", opp: "BOS", shotType: "Layup", zone: "rim", maxShots: 1000 };
const fetchMock = vi.fn();
beforeEach(() => { vi.clearAllMocks(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("shot heatmap route and export contract", () => {
  it("sends the current endpoint's required our parameter and all chosen filters once", async () => {
    fetchMock.mockResolvedValue(Response.json({ image_base64: "image", n_shots: 1000, n_shots_total: 1200, n_shots_rendered: 1000 }));
    const result = await fetchShotHeatmap(filters);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe("/pbp/viz/shot-heatmap");
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual({ season: "2024-25", our: "TOR", opp: "BOS", shot_type: "Layup", zone: "rim", max_shots: "1000" });
    expect(options.headers.Authorization).toBe("Bearer test-session");
    expect(result).toMatchObject({ n_shots: 1000, n_shots_total: 1200, n_shots_rendered: 1000 });
  });

  it.each([400, 401, 403, 404, 422, 503])("does not replace a real HTTP %s error with a legacy blank image", async status => {
    fetchMock.mockResolvedValueOnce(Response.json({ detail: { message: "No shots matched these heatmap filters." } }, { status }));
    fetchMock.mockResolvedValue(Response.json({ image_base64: "legacy blank court" }));
    await expect(fetchShotHeatmap(filters)).rejects.toThrow(status === 404 ? "No shots match" : /.+/);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("does not present an invalid response or network failure as a heatmap", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({ caption: "No image" }));
    await expect(fetchShotHeatmap(filters)).rejects.toThrow("did not return an image");
    fetchMock.mockRejectedValueOnce(new TypeError("network failed"));
    await expect(fetchShotHeatmap(filters)).rejects.toThrow("Could not reach");
  });

  it("includes the same sample cap and filters in the shot-plan PDF", () => {
    const url = new URL(getShotPlanPdfUrl({ ...filters, our: filters.team, k: 3, wOff: .4 }));
    expect(Object.fromEntries(url.searchParams)).toEqual({ season: "2024-25", our: "TOR", opp: "BOS", k: "3", w_off: "0.4", shot_type: "Layup", zone: "rim", max_shots: "1000" });
    expect(new URL(getShotPlanPdfUrl({ season: "2024-25", our: "TOR", opp: "BOS" })).searchParams.get("max_shots")).toBe("35000");
  });
});
