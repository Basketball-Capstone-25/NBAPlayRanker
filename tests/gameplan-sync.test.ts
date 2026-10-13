import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));

import {
  EMPTY_ROLES,
  loadGameplan,
  readLegacyLocal,
  saveGameplan,
  type GameplanBackend,
  type GameplanState,
} from "../app/infrastructure/gameplan-sync";

const M = { season: "2024-25", ourTeam: "TOR", oppTeam: "BOS" };
const STATE: GameplanState = { notes: { pnr: "Slip if they switch" }, plan: [{ id: "pnr", label: "P&R" }], roles: { ...EMPTY_ROLES, ballHandler: "Barnes" } };

function backend(overrides: Partial<GameplanBackend> = {}): GameplanBackend {
  return {
    userId: vi.fn(async () => "coach-1"),
    load: vi.fn(async () => null),
    insert: vi.fn(async () => 1),
    update: vi.fn(async () => 2),
    ...overrides,
  };
}

describe("gameplan sync", () => {
  it("loads the server copy for the signed-in coach", async () => {
    const b = backend({ load: vi.fn(async () => ({ state: STATE, revision: 4 })) });
    expect(await loadGameplan(b, M)).toEqual({ status: "saved", state: STATE, revision: 4, userId: "coach-1" });
  });

  it("falls back to this device when there is no coach session or the server is down", async () => {
    expect(await loadGameplan(backend({ userId: vi.fn(async () => null) }), M)).toMatchObject({ status: "local" });
    expect(await loadGameplan(backend({ load: vi.fn(async () => { throw new Error("offline"); }) }), M)).toMatchObject({ status: "local" });
  });

  it("inserts a new matchup and updates an existing one with its revision", async () => {
    const b = backend();
    expect(await saveGameplan(b, M, STATE, null)).toEqual({ status: "saved", revision: 1 });
    expect(await saveGameplan(b, M, STATE, 1)).toEqual({ status: "saved", revision: 2 });
    expect(b.update).toHaveBeenCalledWith("coach-1", M, STATE, 1);
  });

  it("reports a conflict and returns the latest copy when the revision is stale", async () => {
    const latest = { ...STATE, notes: { pnr: "From the other tab" } };
    const b = backend({ update: vi.fn(async () => null), load: vi.fn(async () => ({ state: latest, revision: 7 })) });
    expect(await saveGameplan(b, M, STATE, 3)).toEqual({ status: "conflict", state: latest, revision: 7 });
  });

  it("treats a duplicate insert from another tab as a conflict", async () => {
    const b = backend({
      insert: vi.fn(async () => { throw Object.assign(new Error("dup"), { code: "23505" }); }),
      load: vi.fn(async () => ({ state: STATE, revision: 1 })),
    });
    expect(await saveGameplan(b, M, STATE, null)).toMatchObject({ status: "conflict", revision: 1 });
  });

  it("returns an error result instead of throwing when the save fails", async () => {
    const b = backend({ update: vi.fn(async () => { throw new Error("network"); }) });
    expect(await saveGameplan(b, M, STATE, 1)).toEqual({ status: "error", message: "network" });
  });

  it("reads legacy local notes only when they contain something", () => {
    const store = new Map<string, string>([["nbaPlayRanker_gameplan_notes_v2", JSON.stringify(STATE.notes)]]);
    expect(readLegacyLocal({ getItem: (k: string) => store.get(k) ?? null })?.notes).toEqual(STATE.notes);
    expect(readLegacyLocal({ getItem: () => null })).toBeNull();
    expect(readLegacyLocal({ getItem: () => "{not json" })).toBeNull();
  });
});
