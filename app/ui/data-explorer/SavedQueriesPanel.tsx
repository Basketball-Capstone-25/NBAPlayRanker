"use client";

// UI Layer — Data Explorer UI: saved queries (SCRUM-771).
// The analyst saves the current dataset, filters and sorting under a name that
// states the question it answers, and opens it again later on any device.

import React, { FormEvent, useEffect, useMemo, useState } from "react";
import type { FilterOptions, QueryDefinition, SavedQuery } from "../../domain/saved-query";
import { createSavedQueryService, type SavedQueryService } from "../../services/saved-queries";

type Status = "loading" | "ready" | "unavailable";

export type SavedQueriesPanelProps = {
  /** The view on screen right now. */
  definition: QueryDefinition;
  /** Values each filter can take in the loaded data, used when a query is opened. */
  options: FilterOptions;
  onOpen(query: SavedQuery): void;
  service?: SavedQueryService;
};

const byName = (a: SavedQuery, b: SavedQuery) => a.name.localeCompare(b.name);

export function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export default function SavedQueriesPanel({ definition, options, onOpen, service: injected }: SavedQueriesPanelProps) {
  const service = useMemo(() => injected ?? createSavedQueryService(), [injected]);
  const [queries, setQueries] = useState<SavedQuery[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [name, setName] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    service.list()
      .then((rows) => { if (alive) { setQueries([...rows].sort(byName)); setStatus("ready"); } })
      .catch(() => { if (alive) setStatus("unavailable"); });
    return () => { alive = false; };
  }, [service]);

  async function save(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const saved = await service.save(name, definition, queries);
      setQueries((list) => [...list, saved].sort(byName));
      setName("");
      setMessage(`Saved "${saved.name}".`);
    } catch (error) {
      setMessage(errorMessage(error, "The query could not be saved. Try again."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card" aria-label="Saved queries" style={{ marginTop: 14 }}>
      <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>Saved queries</h2>
      <p className="muted" style={{ fontSize: 13, margin: 0 }}>
        Save the current dataset, filters and sorting under the question they answer, then open them again on any device.
      </p>

      {status === "unavailable" ? (
        <p role="alert" className="muted" style={{ fontSize: 13 }}>Saved queries are unavailable. Sign in with an analyst account and try again.</p>
      ) : (
        <form onSubmit={save} style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "end", marginTop: 10 }}>
          <label style={{ fontSize: 13, display: "grid", gap: 6, flex: "1 1 320px" }}>
            Question this query answers
            <input
              className="input"
              value={name}
              maxLength={160}
              placeholder="e.g. Our 5 best play types by PPP this season"
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <button className="btn" type="submit" disabled={busy || status !== "ready"}>
            {busy ? "Saving…" : "Save query"}
          </button>
        </form>
      )}

      {message ? <p role="status" style={{ fontSize: 13 }}>{message}</p> : null}

      {status === "loading" ? <p className="muted" style={{ fontSize: 13 }}>Loading saved queries…</p> : null}
      {status === "ready" && queries.length === 0 ? (
        <p className="muted" style={{ fontSize: 13 }}>No saved queries yet.</p>
      ) : null}
      {queries.length > 0 ? (
        <ul aria-label="Your saved queries" style={{ listStyle: "none", padding: 0, margin: "10px 0 0", display: "grid", gap: 8 }}>
          {queries.map((q) => (
            <li key={q.id} style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <strong style={{ flex: "1 1 240px" }}>{q.name}</strong>
              <span className="muted" style={{ fontSize: 12 }}>
                {q.dataset === "shots" ? "Play-by-play shots" : "Synergy play types"}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
