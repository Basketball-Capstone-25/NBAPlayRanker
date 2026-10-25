"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchDatasetCatalog } from "../../infrastructure/api-client";
import type { DatasetMetadata } from "../../infrastructure/dataset-catalog";

export default function DatasetMetadataPanel() {
  const [datasets, setDatasets] = useState<DatasetMetadata[] | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setDatasets(null);
    fetchDatasetCatalog(controller.signal).then(setDatasets).catch(() => {
      if (!controller.signal.aborted) setError("Dataset metadata unavailable. Try again.");
    });
    return () => controller.abort();
  }, [attempt]);
  const active = datasets?.find(d => d.id === "synergy");
  return <section className="card" style={{ padding: 20, marginBottom: 16 }} aria-labelledby="dataset-details-heading">
    <h2 id="dataset-details-heading">Data source</h2>
    {error ? <p role="alert">{error} <button className="btn" onClick={() => setAttempt(x => x + 1)}>Retry</button></p> :
      !datasets ? <p role="status">Loading dataset details…</p> :
      !active || active.status === "unavailable" ? <p role="status">Dataset metadata unavailable.</p> :
      <dl>
        <dt>Source</dt><dd>{active.source}</dd>
        <dt>Seasons covered</dt><dd>{active.seasons.join(", ") || "Unavailable"}</dd>
        <dt>Source rows</dt><dd>{active.row_count?.toLocaleString() ?? "Unavailable"}</dd>
      </dl>}
    {active && active.status !== "unavailable" && <details>
      <summary>Source details</summary>
      <p>The preview combines player rows into team totals.</p>
      <dl><dt>Original pull date</dt><dd>{active.pulled_at || "Not recorded for this snapshot"}</dd>
        <dt>File version</dt><dd style={{ overflowWrap: "anywhere" }}><code>{active.version || "Unavailable"}</code></dd>
      </dl>
    </details>}
    <Link href="/data-dictionary">Open Data Dictionary</Link>
  </section>;
}
