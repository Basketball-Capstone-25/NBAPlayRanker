import { beforeEach, describe, expect, it } from "vitest";
import type { QueryDefinition, SavedQuery } from "../app/domain/saved-query";
import type { SavedQueryBackend } from "../app/infrastructure/saved-query-store";
import { createSavedQueryService, QueryNameError, QueryNameTakenError } from "../app/services/saved-queries";

const definition: QueryDefinition = {
  dataset: "synergy",
  filters: { season: "2024-25", team: "TOR", side: "offense", playType: "", minPoss: 0, search: "" },
  sortKey: "PPP",
  sortDir: "desc",
  rowLimit: 5,
};

/** In-memory stand-in for one analyst's saved_queries rows, with the same unique-name rule. */
function memoryBackend() {
  const rows: SavedQuery[] = [];
  const calls: string[] = [];
  const key = (n: string) => n.trim().replace(/\s+/g, " ").toLowerCase();
  const backend: SavedQueryBackend = {
    async list() { calls.push("list"); return rows.map((r) => ({ ...r })); },
    async insert(name, d) {
      calls.push("insert");
      if (rows.some((r) => key(r.name) === key(name))) throw new QueryNameTakenError(name);
      const row = { ...d, id: `q${rows.length + 1}`, name, updatedAt: "2026-11-09T00:00:00Z" };
      rows.push(row);
      return { ...row };
    },
    async rename(id, name) {
      calls.push("rename");
      const row = rows.find((r) => r.id === id)!;
      row.name = name;
      return { ...row };
    },
    async remove(id) { calls.push("remove"); rows.splice(rows.findIndex((r) => r.id === id), 1); },
  };
  return { backend, rows, calls };
}

let mem: ReturnType<typeof memoryBackend>;
beforeEach(() => { mem = memoryBackend(); });

describe("saved query service (SCRUM-771)", () => {
  it("saves the dataset, every filter, the sort and the row count under a tidied name", async () => {
    const service = createSavedQueryService(mem.backend);
    const saved = await service.save("  Our 5 best play types   by PPP ", definition, []);
    expect(saved.name).toBe("Our 5 best play types by PPP");
    expect(mem.rows[0]).toMatchObject({ dataset: "synergy", filters: definition.filters, sortKey: "PPP", sortDir: "desc", rowLimit: 5 });
    expect(await service.list()).toHaveLength(1);
  });

  it("stops a blank or repeated name before calling Supabase", async () => {
    const service = createSavedQueryService(mem.backend);
    const first = await service.save("Top 5 by PPP", definition, []);
    await expect(service.save(" ", definition, [first])).rejects.toBeInstanceOf(QueryNameError);
    await expect(service.save("top 5 BY ppp", definition, [first])).rejects.toBeInstanceOf(QueryNameError);
    expect(mem.calls.filter((c) => c === "insert")).toHaveLength(1);
  });

  it("reports a name taken on another device even when the local list is out of date", async () => {
    const service = createSavedQueryService(mem.backend);
    await service.save("Top 5 by PPP", definition, []);
    await expect(service.save("Top 5 by PPP", definition, [])).rejects.toBeInstanceOf(QueryNameTakenError);
  });

  it("keeps the row count within 1 to 10,000", async () => {
    const service = createSavedQueryService(mem.backend);
    await service.save("Everything", { ...definition, rowLimit: 50000 }, []);
    await service.save("Nothing", { ...definition, rowLimit: 0 }, mem.rows);
    expect(mem.rows.map((r) => r.rowLimit)).toEqual([10000, 1]);
  });

  it("renames and deletes a saved query", async () => {
    const service = createSavedQueryService(mem.backend);
    const saved = await service.save("Top 5 by PPP", definition, []);
    const renamed = await service.rename(saved.id, "Top 5 play types by PPP, 2024-25", mem.rows);
    expect(renamed.name).toBe("Top 5 play types by PPP, 2024-25");
    await service.remove(saved.id);
    expect(await service.list()).toEqual([]);
  });

  it("does not let a rename take another query's name", async () => {
    const service = createSavedQueryService(mem.backend);
    const a = await service.save("Top 5 by PPP", definition, []);
    await service.save("Worst 5 by PPP", definition, mem.rows);
    await expect(service.rename(a.id, "worst 5 by ppp", mem.rows)).rejects.toBeInstanceOf(QueryNameError);
  });
});
