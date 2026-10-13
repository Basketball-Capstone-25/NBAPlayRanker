/**
 * Gameplan sync (SCRUM-483 on top of the SCRUM-481 storage contract).
 *
 * Loads and saves the coach's notes, plan and roles for one directed matchup
 * in Supabase, and falls back to this device's localStorage when Supabase or a
 * coach session is unavailable. Legacy local notes are never uploaded silently:
 * the UI offers an explicit import.
 */

import { createClient } from "../../lib/supabase/client";

export type PlanItem = { id: string; label: string };
export type Roles = { ballHandler: string; screener: string; cornerSpacer: string; cutter: string; safety: string };
export type GameplanState = { notes: Record<string, string>; plan: PlanItem[]; roles: Roles };
export type Matchup = { season: string; ourTeam: string; oppTeam: string };

/** loading → saved | local | error; saving → saved | conflict | local | error */
export type SyncStatus = "loading" | "saving" | "saved" | "local" | "error" | "conflict";

export const LEGACY_KEYS = {
  notes: "nbaPlayRanker_gameplan_notes_v2",
  plan: "nbaPlayRanker_gameplan_plan_v2",
  roles: "nbaPlayRanker_gameplan_roles_v2",
} as const;

export const EMPTY_ROLES: Roles = { ballHandler: "", screener: "", cornerSpacer: "", cutter: "", safety: "" };

export interface GameplanBackend {
  /** Signed-in user id, or null when there is no usable session. */
  userId(): Promise<string | null>;
  load(userId: string, m: Matchup): Promise<{ state: GameplanState; revision: number } | null>;
  insert(m: Matchup, s: GameplanState): Promise<number>;
  /** Returns the new revision, or null when the revision no longer matches. */
  update(userId: string, m: Matchup, s: GameplanState, revision: number): Promise<number | null>;
}

export type LoadResult =
  | { status: "saved"; state: GameplanState | null; revision: number | null; userId: string }
  | { status: "local"; reason: string };

export type SaveResult =
  | { status: "saved"; revision: number }
  | { status: "conflict"; state: GameplanState | null; revision: number | null }
  | { status: "local"; reason: string }
  | { status: "error"; message: string };

export function hasContent(s: GameplanState | null): boolean {
  if (!s) return false;
  return Object.values(s.notes).some((v) => v.trim()) || s.plan.length > 0 || Object.values(s.roles).some((v) => v.trim());
}

export function readLegacyLocal(storage: Pick<Storage, "getItem"> | null): GameplanState | null {
  if (!storage) return null;
  const parse = <T,>(key: string, fallback: T): T => {
    try {
      const raw = storage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : fallback;
    } catch {
      return fallback;
    }
  };
  const state: GameplanState = {
    notes: parse(LEGACY_KEYS.notes, {}),
    plan: parse(LEGACY_KEYS.plan, []),
    roles: { ...EMPTY_ROLES, ...parse(LEGACY_KEYS.roles, {}) },
  };
  return hasContent(state) ? state : null;
}

export async function loadGameplan(backend: GameplanBackend, m: Matchup): Promise<LoadResult> {
  try {
    const userId = await backend.userId();
    if (!userId) return { status: "local", reason: "Not signed in as a coach" };
    const row = await backend.load(userId, m);
    return { status: "saved", state: row?.state ?? null, revision: row?.revision ?? null, userId };
  } catch {
    return { status: "local", reason: "Server unavailable" };
  }
}

export async function saveGameplan(
  backend: GameplanBackend,
  m: Matchup,
  state: GameplanState,
  revision: number | null,
): Promise<SaveResult> {
  try {
    const userId = await backend.userId();
    if (!userId) return { status: "local", reason: "Not signed in as a coach" };
    if (revision === null) {
      try {
        return { status: "saved", revision: await backend.insert(m, state) };
      } catch (e: any) {
        if (e?.code !== "23505") throw e; // another tab created it first
        const fresh = await backend.load(userId, m);
        return { status: "conflict", state: fresh?.state ?? null, revision: fresh?.revision ?? null };
      }
    }
    const next = await backend.update(userId, m, state, revision);
    if (next !== null) return { status: "saved", revision: next };
    const fresh = await backend.load(userId, m);
    return { status: "conflict", state: fresh?.state ?? null, revision: fresh?.revision ?? null };
  } catch (e: any) {
    return { status: "error", message: e?.message || "Sync failed" };
  }
}

const COLUMNS = "notes,plan,roles,revision";
const identity = (m: Matchup) => ({ season: m.season, our_team: m.ourTeam, opp_team: m.oppTeam });
const toState = (row: any): GameplanState => ({
  notes: row.notes ?? {},
  plan: row.plan ?? [],
  roles: { ...EMPTY_ROLES, ...(row.roles ?? {}) },
});

export function supabaseGameplanBackend(): GameplanBackend {
  const supabase = createClient();
  return {
    async userId() {
      const { data } = await supabase.auth.getSession();
      return data.session?.user?.id ?? null;
    },
    async load(userId, m) {
      const { data, error } = await supabase.from("gameplan_notes").select(COLUMNS)
        .eq("user_id", userId).eq("season", m.season).eq("our_team", m.ourTeam).eq("opp_team", m.oppTeam)
        .maybeSingle();
      if (error) throw error;
      return data ? { state: toState(data), revision: Number((data as any).revision) } : null;
    },
    async insert(m, s) {
      const { data, error } = await supabase.from("gameplan_notes")
        .insert({ ...identity(m), notes: s.notes, plan: s.plan, roles: s.roles })
        .select(COLUMNS).single();
      if (error) throw error;
      return Number((data as any).revision);
    },
    async update(userId, m, s, revision) {
      const { data, error } = await supabase.from("gameplan_notes")
        .update({ notes: s.notes, plan: s.plan, roles: s.roles })
        .eq("user_id", userId).eq("season", m.season).eq("our_team", m.ourTeam).eq("opp_team", m.oppTeam)
        .eq("revision", revision).select(COLUMNS).maybeSingle();
      if (error) throw error;
      return data ? Number((data as any).revision) : null;
    },
  };
}

export const SYNC_LABELS: Record<SyncStatus, string> = {
  loading: "Loading saved gameplan…",
  saving: "Saving…",
  saved: "Saved",
  local: "Saved on this device only",
  error: "Sync failed",
  conflict: "Updated elsewhere — reloaded the latest copy",
};
