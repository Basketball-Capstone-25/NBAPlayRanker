/**
 * Infrastructure Layer — saved Data Explorer queries in Supabase (SCRUM-771).
 *
 * Reads and writes the signed-in analyst's rows in public.saved_queries.
 * Row level security limits every call to the analyst's own rows, so this
 * adapter never filters by user id on the client.
 */

import { createClient } from "../../lib/supabase/client";
import type { FilterValues, QueryDefinition, SavedQuery, SavedQueryDataset, SortDirection } from "../domain/saved-query";

export interface SavedQueryBackend {
  list(): Promise<SavedQuery[]>;
  insert(name: string, definition: QueryDefinition): Promise<SavedQuery>;
  rename(id: string, name: string): Promise<SavedQuery>;
  remove(id: string): Promise<void>;
}

/** The name is already used by this analyst (unique index on saved_queries). */
export class QueryNameTakenError extends Error {
  constructor(name: string) {
    super(`You already have a query called "${name}".`);
    this.name = "QueryNameTakenError";
  }
}

type Row = {
  id: string;
  name: string;
  dataset: SavedQueryDataset;
  filters: FilterValues | null;
  sort_key: string | null;
  sort_dir: SortDirection | null;
  row_limit: number;
  updated_at: string;
};

const COLUMNS = "id,name,dataset,filters,sort_key,sort_dir,row_limit,updated_at";

export function toSavedQuery(row: Row): SavedQuery {
  return {
    id: row.id,
    name: row.name,
    dataset: row.dataset,
    filters: row.filters ?? {},
    sortKey: row.sort_key,
    sortDir: row.sort_dir,
    rowLimit: Number(row.row_limit),
    updatedAt: row.updated_at,
  };
}

function fail(error: { code?: string; message?: string }, name: string): never {
  if (error.code === "23505") throw new QueryNameTakenError(name);
  throw new Error(error.message || "Saved queries are unavailable right now.");
}

export function supabaseSavedQueryStore(): SavedQueryBackend {
  const supabase = createClient();
  return {
    async list() {
      const { data, error } = await supabase.from("saved_queries").select(COLUMNS).order("name");
      if (error) fail(error, "");
      return ((data ?? []) as Row[]).map(toSavedQuery);
    },
    async insert(name, d) {
      const { data, error } = await supabase.from("saved_queries")
        .insert({ name, dataset: d.dataset, filters: d.filters, sort_key: d.sortKey, sort_dir: d.sortDir, row_limit: d.rowLimit })
        .select(COLUMNS).single();
      if (error) fail(error, name);
      return toSavedQuery(data as Row);
    },
    async rename(id, name) {
      const { data, error } = await supabase.from("saved_queries").update({ name }).eq("id", id).select(COLUMNS).single();
      if (error) fail(error, name);
      return toSavedQuery(data as Row);
    },
    async remove(id) {
      const { error } = await supabase.from("saved_queries").delete().eq("id", id);
      if (error) fail(error, "");
    },
  };
}
