// app/services/saved-queries.ts
//
// Application Layer — Frontend Services: saved Data Explorer queries (SCRUM-771).
// The facade the Data Explorer calls. It checks the name before anything is
// sent, so a blank or repeated name never reaches Supabase.

import {
  MAX_ROW_LIMIT,
  normalizeQueryName,
  validateName,
  type QueryDefinition,
  type SavedQuery,
} from "../domain/saved-query";
import { supabaseSavedQueryStore, type SavedQueryBackend } from "../infrastructure/saved-query-store";

export { QueryNameTakenError } from "../infrastructure/saved-query-store";

/** A name the analyst has to change before the query can be saved. */
export class QueryNameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QueryNameError";
  }
}

export type SavedQueryService = {
  list(): Promise<SavedQuery[]>;
  save(name: string, definition: QueryDefinition, existing: readonly SavedQuery[]): Promise<SavedQuery>;
  rename(id: string, name: string, existing: readonly SavedQuery[]): Promise<SavedQuery>;
  remove(id: string): Promise<void>;
};

function checkedName(name: string, existing: readonly SavedQuery[], ignoreId?: string): string {
  const problem = validateName(name, existing, ignoreId);
  if (problem) throw new QueryNameError(problem);
  return normalizeQueryName(name);
}

function rowLimit(limit: number): number {
  if (!Number.isFinite(limit)) return 200;
  return Math.min(MAX_ROW_LIMIT, Math.max(1, Math.round(limit)));
}

export function createSavedQueryService(backend: SavedQueryBackend = supabaseSavedQueryStore()): SavedQueryService {
  return {
    list: () => backend.list(),
    // async so a bad name comes back as a rejected promise, like a Supabase error.
    save: async (name, definition, existing) =>
      backend.insert(checkedName(name, existing), { ...definition, rowLimit: rowLimit(definition.rowLimit) }),
    rename: async (id, name, existing) => backend.rename(id, checkedName(name, existing, id)),
    remove: (id) => backend.remove(id),
  };
}
