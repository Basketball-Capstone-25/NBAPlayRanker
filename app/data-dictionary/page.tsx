"use client";

// Route /data-dictionary — searchable definitions of every dataset column.

import Link from "next/link";
import styles from "../reference.module.css";
import { useEffect, useMemo, useState } from "react";
import { fetchDataDictionary } from "../infrastructure/api-client";
import type { DictionaryColumn } from "../infrastructure/dataset-catalog";

const countLabel = (count: number, noun: string) => `${count} ${noun}${count === 1 ? "" : "s"}`;

export default function DataDictionaryPage() {
  const [columns, setColumns] = useState<DictionaryColumn[] | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [dataset, setDataset] = useState("All datasets");
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetchDataDictionary(controller.signal).then(data => setColumns(data.columns)).catch(() => {
      if (!controller.signal.aborted) setError("Data Dictionary unavailable. Please try again.");
    });
    return () => controller.abort();
  }, [attempt]);
  const visible = useMemo(() => (columns || []).filter(c =>
    (dataset === "All datasets" || c.dataset_name === dataset) &&
    `${c.dataset_name} ${c.column} ${c.meaning}`.toLowerCase().includes(query.toLowerCase())), [columns, query, dataset]);
  const datasets = [...new Set((columns || []).map(c => c.dataset_name))].sort();
  const undocumented = visible.filter(c => !c.documented).length;
  return <section className={styles.page}>
    <header className={styles.heading}><div><p className={styles.eyebrow}>REFERENCE</p><h1>Data Dictionary</h1><p>Look up column definitions, units and data sources.</p></div><Link className="btn" href="/glossary">Glossary ↗</Link></header>
    <div className={styles.filters}>
      <label htmlFor="dictionary-search">Find a column<input className="input" type="search" id="dictionary-search" placeholder="Search names or meanings" value={query} onChange={event => setQuery(event.target.value)} /></label>
      <label>Dataset<select className="input" value={dataset} onChange={e => setDataset(e.target.value)}><option>All datasets</option>{datasets.map(d => <option key={d}>{d}</option>)}</select></label>
      <button className="btn" onClick={() => { setQuery(""); setDataset("All datasets"); }}>Clear</button>
    </div>
    {error ? <p role="alert">{error} <button className="btn" onClick={() => setAttempt(x => x + 1)}>Retry</button></p> :
      columns === null ? <p role="status">Loading column definitions…</p> : <>
        <p className={styles.count} role="status">{countLabel(visible.length, "column")} shown.{undocumented > 0 && ` ${countLabel(undocumented, "column")} still need a definition.`}</p>
        <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Column definitions"><table>
          <thead><tr><th scope="col">Dataset</th><th scope="col">Column</th><th scope="col">Meaning</th><th scope="col">Unit</th><th scope="col">Source</th></tr></thead>
          <tbody>{visible.map(c => <tr key={`${c.dataset}:${c.column}`}>
            <td>{c.dataset_name}</td><th scope="row"><code>{c.column}</code></th>
            <td>{c.documented ? c.meaning : <strong>Undocumented column</strong>}</td><td>{c.unit}</td><td>{c.source}</td>
          </tr>)}</tbody>
        </table></div>
        {visible.length === 0 && <p>No matching columns.</p>}
      </>}
  </section>;
}
