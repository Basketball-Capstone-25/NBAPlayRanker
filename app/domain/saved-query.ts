// Domain Layer — saved Data Explorer queries (SCRUM-771).
// A saved query is the analyst's current filters and sorting stored under a
// name that states the question it answers. These rules are pure so the panel,
// the service and the tests all use them the same way.

export type SavedQueryDataset = "synergy" | "shots";
export type SortDirection = "asc" | "desc";

/** Filter values as they appear in the Data Explorer controls. Blank means "all". */
export type FilterValues = Record<string, string | number>;

export type QueryDefinition = {
  dataset: SavedQueryDataset;
  filters: FilterValues;
  sortKey: string | null;
  sortDir: SortDirection | null;
  rowLimit: number;
};

export type SavedQuery = QueryDefinition & {
  id: string;
  name: string;
  updatedAt: string;
};

export const MAX_QUERY_NAME = 120;
export const MAX_ROW_LIMIT = 10000;

export function normalizeQueryName(name: string): string {
  return name.trim().replace(/\s+/g, " ");
}

/**
 * Returns a message when the name cannot be used, or "" when it can.
 * Names are compared without case or extra spaces, the same rule the
 * saved_queries table enforces for each analyst.
 */
export function validateName(name: string, existing: readonly SavedQuery[], ignoreId?: string): string {
  const clean = normalizeQueryName(name);
  if (!clean) return "Give the query a name, for example the question it answers.";
  if (clean.length > MAX_QUERY_NAME) return `Keep the name to ${MAX_QUERY_NAME} characters or fewer.`;
  const key = clean.toLowerCase();
  const taken = existing.some((q) => q.id !== ignoreId && normalizeQueryName(q.name).toLowerCase() === key);
  return taken ? `You already have a query called "${clean}".` : "";
}
