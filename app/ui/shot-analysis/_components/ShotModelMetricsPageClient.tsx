"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ShotLaterSeasonResults, shotModelName } from "./ShotLaterSeasonResults";
import { fetchShotModelMetrics } from "../../../services/shotAnalysis";
import type { ShotLaterSeason } from "../../../domain/shot-analysis";

type MetricsRow = {
  model: string;
  RMSE_mean: number; RMSE_std: number;
  MAE_mean: number; MAE_std: number;
  R2_mean: number; R2_std: number;
};

type MetricsResponse = {
  source_sha256?: string;
  best_model?: string;
  later_season?: ShotLaterSeason;
  n_splits: number;
  metrics: MetricsRow[];
};

const fmt = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? "Unavailable" : value.toFixed(3);

export function ShotMetricsContent({ data, loading, error, onReload }: {
  data: MetricsResponse | null; loading: boolean; error: string | null; onReload: () => void;
}) {
  const [copyStatus, setCopyStatus] = useState("");
  const rows = !loading && !error && Array.isArray(data?.metrics) ? data.metrics : [];
  const best = [...rows].filter(row => Number.isFinite(row.RMSE_mean)).sort((a, b) => a.RMSE_mean - b.RMSE_mean)[0];
  async function copySummary() {
    if (!best) return;
    try {
      await navigator.clipboard.writeText(`Training cross-validation: ${shotModelName(best.model)} — RMSE ${fmt(best.RMSE_mean)} ± ${fmt(best.RMSE_std)}, MAE ${fmt(best.MAE_mean)} ± ${fmt(best.MAE_std)}, R² ${fmt(best.R2_mean)} ± ${fmt(best.R2_std)}.`);
      setCopyStatus("Summary copied.");
    } catch {
      setCopyStatus("Could not copy the summary.");
    }
  }
  return <>
    <header className="card">
      <h1 className="h1">Shot Model Metrics</h1>
      <p>Compare how well the models estimate points for a proposed shot.</p>
      <p className="muted">Shot location and type must be supplied. These results do not measure pre-possession forecasts.</p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <Link href="/shot-plan">Shot Plan</Link>
        <Link href="/shot-statistical-analysis">Shot Statistical Analysis</Link>
        <button className="btn btn--secondary" disabled={loading} onClick={onReload}>{loading ? "Loading results…" : "Reload results"}</button>
      </div>
    </header>
    {loading ? <p role="status" className="card">Loading saved results…</p> : error ?
      <section className="card" role="alert"><h2>Results unavailable</h2><p>{error}</p></section> : rows.length ? <>
        {data?.later_season && <ShotLaterSeasonResults result={data.later_season} selectedModel={data.best_model} />}
        <section className="card">
          <h2>Training-season cross-validation</h2>
          {best && <p>{shotModelName(best.model)} had the lowest average prediction error across {data?.n_splits} groups of games.</p>}
          <p className="muted">RMSE and MAE measure error in points per shot; lower is better. R² measures improvement over predicting the mean; higher is better.</p>
          <div style={{ overflowX: "auto" }}><table className="table">
            <thead><tr><th scope="col">Model</th><th scope="col">RMSE</th><th scope="col">MAE</th><th scope="col">R²</th></tr></thead>
            <tbody>{rows.map(row => <tr key={row.model}><th scope="row">{shotModelName(row.model)}</th><td>{fmt(row.RMSE_mean)}</td><td>{fmt(row.MAE_mean)}</td><td>{fmt(row.R2_mean)}</td></tr>)}</tbody>
          </table></div>
          <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
            {data?.n_splits === 5 && /^[a-f0-9]{64}$/.test(data.source_sha256 ?? "") &&
              <a className="btn btn--secondary" href={`/api/exports/shot-metrics?version=${data.source_sha256}`} target="_blank" rel="noopener noreferrer">Export training results (CSV)</a>}
            <button className="btn btn--secondary" disabled={!best} onClick={copySummary}>Copy training summary</button>
          </div>
          <p role="status">{copyStatus}</p>
          <details>
            <summary>Evaluation details</summary>
            <p>Saved results use five folds grouped by game. Model fitting runs separately; reloading this page only retrieves the saved evaluation.</p>
            <p>Mean ± standard deviation across training folds:</p>
            <div style={{ overflowX: "auto" }}><table className="table">
              <thead><tr><th scope="col">Model</th><th scope="col">RMSE</th><th scope="col">MAE</th><th scope="col">R²</th></tr></thead>
              <tbody>{rows.map(row => <tr key={row.model}><th scope="row">{shotModelName(row.model)}</th><td>{fmt(row.RMSE_mean)} ± {fmt(row.RMSE_std)}</td><td>{fmt(row.MAE_mean)} ± {fmt(row.MAE_std)}</td><td>{fmt(row.R2_mean)} ± {fmt(row.R2_std)}</td></tr>)}</tbody>
            </table></div>
            {data?.source_sha256 && <p style={{ overflowWrap: "anywhere" }}>Shot data version: <code>sha256:{data.source_sha256}</code></p>}
          </details>
        </section>
      </> : <section className="card"><p>No saved results are available for this dataset.</p></section>}
  </>;
}

export default function ShotModelMetricsPage() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    setLoading(true); setError(null); setData(null);
    fetchShotModelMetrics(5).then(result => { if (active) setData(result); }).catch(reason => {
      if (active) setError(reason instanceof Error ? reason.message : "The saved results could not be loaded.");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [attempt]);
  return <main className="page" style={{ paddingBottom: 56 }}><ShotMetricsContent data={data} loading={loading} error={error} onReload={() => {
    setLoading(true); setData(null); setError(null); setAttempt(value => value + 1);
  }} /></main>;
}
