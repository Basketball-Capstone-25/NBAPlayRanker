"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { fetchDataDictionary } from "../infrastructure/api-client";
import type { DictionaryColumn } from "../infrastructure/dataset-catalog";

export default function DataDictionaryPage() {
  const [columns, setColumns] = useState<DictionaryColumn[] | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetchDataDictionary(controller.signal).then(data => setColumns(data.columns)).catch(() => {
      if (!controller.signal.aborted) setError("Data Dictionary unavailable. Please try again.");
    });
    return () => controller.abort();
  }, [attempt]);
  const visible = useMemo(() => (columns || []).filter(c =>
    `${c.dataset_name} ${c.column} ${c.meaning}`.toLowerCase().includes(query.toLowerCase())), [columns, query]);
  return <main className="page" style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
    <h1>Data Dictionary</h1>
    <p>Column names, meanings and units from the files used by the app. Missing definitions are marked as undocumented.</p>
    <p><Link href="/glossary">Glossary</Link> · <Link href="/data-explorer">Data Explorer</Link></p>
    <label htmlFor="dictionary-search">Search columns or datasets</label>{" "}
    <input id="dictionary-search" value={query} onChange={event => setQuery(event.target.value)} />
    {error ? <p role="alert">{error} <button className="btn" onClick={() => setAttempt(x => x + 1)}>Retry</button></p> :
      columns === null ? <p role="status">Loading column definitions…</p> : <>
        <p>{visible.length} columns shown. {columns.filter(c => !c.documented).length} undocumented.</p>
        <div style={{ overflowX: "auto" }}><table className="table">
          <thead><tr><th scope="col">Dataset</th><th scope="col">Column</th><th scope="col">Meaning</th><th scope="col">Unit</th><th scope="col">Source</th></tr></thead>
          <tbody>{visible.map(c => <tr key={`${c.dataset}:${c.column}`}>
            <td>{c.dataset_name}</td><th scope="row"><code>{c.column}</code></th>
            <td>{c.documented ? c.meaning : <strong>Undocumented column</strong>}</td><td>{c.unit}</td><td>{c.source}</td>
          </tr>)}</tbody>
        </table></div>
        {visible.length === 0 && <p>No matching columns.</p>}
      </>}
  </main>;
}
