import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReloadDataset } from "../app/domain/dataset-reload";
import { reloadDataset } from "../app/infrastructure/dataset-reload";

const { getSession, makeClient } = vi.hoisted(() => ({ getSession: vi.fn(), makeClient: vi.fn() }));
vi.mock("../lib/supabase/client", () => ({ createClient: makeClient }));
const report = (dataset: ReloadDataset = "synergy") => ({
  dataset: dataset === "pbp" ? "pbp_shots" : "synergy", status: "reloaded",
  source_file: dataset === "pbp" ? "shots_clean.parquet" : "synergy.csv",
  previous_version: "v1", dataset_version: "v1", changed: false,
  rows: 100, seasons: ["2024-25"], ...(dataset === "synergy" ? { quality_status: "pass" } : {}),
});

describe("dataset reload requests", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.resetAllMocks(); vi.stubGlobal("fetch", fetchMock);
    makeClient.mockReturnValue({ auth: { getSession } });
    getSession.mockResolvedValue({ data: { session: { access_token: "test-token" } }, error: null });
  });
  afterEach(() => vi.unstubAllGlobals());

  it.each(["synergy", "pbp"] as const)("sends one authenticated POST to the fixed %s endpoint", async dataset => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(report(dataset))));
    await expect(reloadDataset(dataset)).resolves.toEqual(report(dataset));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(new URL(url).pathname).toBe(`/data/reload/${dataset}`);
    expect(new URL(url).search).toBe("");
    expect(url).not.toContain("test-token");
    expect(options).toEqual({ method: "POST", cache: "no-store", signal: undefined, headers: { Authorization: "Bearer test-token" } });
  });
  it("rejects unknown dataset names without looking up a session or sending a request", async () => {
    await expect(reloadDataset("../../admin" as ReloadDataset)).rejects.toMatchObject({ outcome: "not_started" });
    expect(getSession).not.toHaveBeenCalled(); expect(fetchMock).not.toHaveBeenCalled();
  });
  it("does not send a POST without a usable session", async () => {
    getSession.mockResolvedValueOnce({ data: { session: null } }).mockRejectedValueOnce(new Error("private detail"))
      .mockResolvedValueOnce({ data: { session: { access_token: "test-token" } }, error: new Error("private detail") });
    for (let request = 0; request < 3; request++) {
      await expect(reloadDataset("synergy")).rejects.toMatchObject({ message: "Sign in again to reload data.", outcome: "not_started" });
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("does not send a POST when the session client cannot start", async () => {
    makeClient.mockImplementationOnce(() => { throw new Error("private configuration detail"); });
    await expect(reloadDataset("synergy")).rejects.toMatchObject({ message: "Sign in again to reload data.", outcome: "not_started" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it.each([
    [401, "Sign in again", "not_started"], [403, "analyst account", "not_started"],
    [404, "not available", "not_started"], [409, "already running", "rejected"],
    [500, "unavailable", "unknown"], [503, "unavailable", "unknown"],
  ])("explains HTTP %s without retrying or showing server text", async (status, message, outcome) => {
    fetchMock.mockResolvedValue(new Response("private server detail", { status: Number(status) }));
    await expect(reloadDataset("synergy")).rejects.toMatchObject({ message: expect.stringContaining(message), outcome, status });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it("shows the checked-file rejection reason", async () => {
    const detail = "Reload failed; the previous data is still in use. shots_clean.parquet is missing columns: GAME_ID.";
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail }), { status: 422 }));
    await expect(reloadDataset("pbp")).rejects.toMatchObject({ message: detail, outcome: "rejected", status: 422 });
  });
  it.each([
    "not JSON", JSON.stringify({ detail: [{ loc: ["body"], msg: "private detail" }] }),
    JSON.stringify({ detail: "<html>server stack</html>" }),
    JSON.stringify({ detail: "Reload failed; the previous data is still in use. <script>bad</script>" }),
  ])("uses a plain rejection for malformed error bodies", async body => {
    fetchMock.mockResolvedValue(new Response(body, { status: 422 }));
    await expect(reloadDataset("pbp")).rejects.toMatchObject({ message: "Reload failed. The previous data is still in use.", outcome: "rejected" });
  });
  it("does not retry a lost response or claim that the old data was retained", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(reloadDataset("synergy")).rejects.toMatchObject({ message: expect.stringContaining("may still be running"), outcome: "unknown" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it("does not report success from invalid JSON or a response for the other dataset", async () => {
    fetchMock.mockResolvedValueOnce(new Response("not json"))
      .mockResolvedValueOnce(new Response(JSON.stringify(report("pbp"))));
    await expect(reloadDataset("synergy")).rejects.toMatchObject({ outcome: "unknown" });
    await expect(reloadDataset("synergy")).rejects.toMatchObject({ outcome: "unknown" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
  it("does not start a reload if canceled before or during session lookup", async () => {
    const controller = new AbortController(); controller.abort();
    await expect(reloadDataset("pbp", controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(getSession).not.toHaveBeenCalled();
    const next = new AbortController();
    getSession.mockImplementationOnce(async () => { next.abort(); return { data: { session: null } }; });
    await expect(reloadDataset("pbp", next.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("treats cancellation after sending as an unknown result", async () => {
    const controller = new AbortController();
    fetchMock.mockImplementationOnce(async () => { controller.abort(); throw controller.signal.reason; });
    await expect(reloadDataset("pbp", controller.signal)).rejects.toMatchObject({ outcome: "unknown" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
