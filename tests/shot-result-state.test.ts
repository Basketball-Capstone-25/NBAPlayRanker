import { describe, expect, it } from "vitest";
import { emptyShotResult, selectShotResult, ShotResultRequest, type ShotRequestState } from "../app/infrastructure/shot-result-state";

const filters = { season: "2025-26", team: "TOR", opp: "BOS", shotType: "", zone: "", maxShots: 35000, k: 5, wOff: .7, limit: 50 };
function deferred<T>() {
  let resolve!: (result: T) => void, reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

describe("shot results stay tied to the submitted inputs", () => {
  it("hides old results and export inputs for every changed filter", async () => {
    let state = emptyShotResult<typeof filters, string>();
    await new ShotResultRequest<typeof filters, string>().run(filters, async () => "shot result", next => { state = next; });
    expect(selectShotResult(state, filters)).toMatchObject({ result: "shot result", inputs: filters, changed: false });
    const changes = { season: "2024-25", team: "LAL", opp: "DEN", shotType: "Layup", zone: "rim", maxShots: 1000, k: 3, wOff: .4, limit: 100 };
    for (const [key, value] of Object.entries(changes)) {
      expect(selectShotResult(state, { ...filters, [key]: value })).toMatchObject({ result: null, inputs: null, changed: true });
    }
    expect(selectShotResult(state, filters).result).toBe("shot result");
  });

  it("clears a previous success during retry and after failure, then accepts recovery", async () => {
    let state = emptyShotResult<typeof filters, string>();
    const request = new ShotResultRequest<typeof filters, string>();
    const publish = (next: ShotRequestState<typeof filters, string>) => { state = next; };
    await request.run(filters, async () => "old result", publish);
    const pending = deferred<string>();
    const run = request.run(filters, () => pending.promise, publish);
    expect(selectShotResult(state, filters)).toMatchObject({ loading: true, result: null, inputs: null });
    pending.reject(new Error("Service unavailable")); await run;
    expect(selectShotResult(state, filters)).toMatchObject({ loading: false, result: null, inputs: null, error: "Service unavailable" });
    await request.run(filters, async () => "recovered", publish);
    expect(selectShotResult(state, filters)).toMatchObject({ result: "recovered", error: null, inputs: filters });
  });

  it("does not show a late response under edited inputs", async () => {
    let state = emptyShotResult<typeof filters, string>();
    const pending = deferred<string>();
    const run = new ShotResultRequest<typeof filters, string>().run(filters, () => pending.promise, next => { state = next; });
    pending.resolve("TOR result"); await run;
    expect(selectShotResult(state, { ...filters, team: "LAL" })).toMatchObject({ result: null, inputs: null, changed: true });
  });

  it("ignores an older completion without stopping the new request", async () => {
    let state = emptyShotResult<typeof filters, string>();
    const request = new ShotResultRequest<typeof filters, string>();
    const publish = (next: ShotRequestState<typeof filters, string>) => { state = next; };
    const old = deferred<string>(), latest = deferred<string>();
    const oldRun = request.run(filters, () => old.promise, publish);
    const nextInputs = { ...filters, team: "LAL" };
    const newRun = request.run(nextInputs, () => latest.promise, publish);
    old.resolve("old TOR result"); await oldRun;
    expect(selectShotResult(state, nextInputs)).toMatchObject({ loading: true, result: null });
    latest.resolve("new LAL result"); await newRun;
    expect(selectShotResult(state, nextInputs)).toMatchObject({ loading: false, result: "new LAL result", inputs: nextInputs });
  });

  it("cancels pending work on reset or unmount", async () => {
    const published: unknown[] = [], pending = deferred<string>();
    const request = new ShotResultRequest<typeof filters, string>();
    const run = request.run(filters, () => pending.promise, next => published.push(next));
    request.cancel(); pending.resolve("late result"); await run;
    expect(published).toHaveLength(1);
  });
});
