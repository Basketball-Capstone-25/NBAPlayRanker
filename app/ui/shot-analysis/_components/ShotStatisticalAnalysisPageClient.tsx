"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { fetchShotMlAnalysis } from "../../../services/shotAnalysis";
import type { ShotAnalysis } from "../../../domain/shot-analysis";

const value = (n: number | null | undefined, digits = 0) => n == null || !Number.isFinite(n) ? "Unavailable" : n.toLocaleString(undefined, { maximumFractionDigits: digits });

export function ShotAnalysisResults({ data }: { data: ShotAnalysis }) {
  return <>
    <section className="card">
      <h2>Dataset</h2>
      <p>{value(data.dataset.rows_used)} sampled shot attempts · {value(data.dataset.n_games)} games · {value(data.dataset.n_seasons)} seasons</p>
      <p>Target: {data.dataset.feature_spec.target ?? "Unavailable"}. Numeric inputs: {data.dataset.feature_spec.numeric.join(", ") || "Unavailable"}.</p>
      <p>These results describe recorded or proposed shots. Shot location and type are not known before a possession, so this is not a pre-possession forecast.</p>
      {data.dataset.notes.map((note, index) => <p key={index} className="muted">{note}</p>)}
    </section>
    <section className="card">
      <h2>Missing values</h2>
      {Object.keys(data.eda.missing_counts).length ? <table className="table"><thead><tr><th>Input</th><th>Missing rows</th></tr></thead><tbody>
        {Object.entries(data.eda.missing_counts).map(([name, count]) => <tr key={name}><td>{name}</td><td>{value(count)}</td></tr>)}
      </tbody></table> : <p>Missing-value counts are unavailable in this saved result.</p>}
    </section>
    <section className="card">
      <h2>Points distribution</h2>
      {data.eda.points.hist.counts.length && data.eda.points.hist.bins.length === data.eda.points.hist.counts.length + 1 ? <table className="table"><thead><tr><th>Points range</th><th>Shot attempts</th></tr></thead><tbody>
        {data.eda.points.hist.counts.map((count, i) => <tr key={i}><td>{value(data.eda.points.hist.bins[i], 2)}–{value(data.eda.points.hist.bins[i + 1], 2)}</td><td>{value(count)}</td></tr>)}
      </tbody></table> : <p>The points distribution is unavailable in this saved result.</p>}
    </section>
    <section className="card">
      <h2>Input correlations with points</h2>
      {data.feature_target_corr.length ? <table className="table"><thead><tr><th>Input</th><th>Correlation</th></tr></thead><tbody>
        {data.feature_target_corr.map(row => <tr key={row.feature}><td>{row.feature}</td><td>{value(row.corr, 4)}</td></tr>)}
      </tbody></table> : <p>Input correlations are unavailable in this saved result.</p>}
    </section>
    <section className="card">
      <h2>Correlation matrix</h2>
      {data.correlations.labels.length ? <div style={{ overflowX: "auto" }}><table className="table">
        <thead><tr><th>Input</th>{data.correlations.labels.map(label => <th key={label}>{label}</th>)}</tr></thead>
        <tbody>{data.correlations.labels.map((label, i) => <tr key={label}><th>{label}</th>{data.correlations.labels.map((column, j) => <td key={column}>{value(data.correlations.matrix[i]?.[j], 3)}</td>)}</tr>)}</tbody>
      </table></div> : <p>The correlation matrix is unavailable in this saved result.</p>}
    </section>
    <section className="card">
      <h2>Feature scores</h2>
      {data.feature_selection ? <>
        <p>{data.feature_selection.method}. Selected input count: {value(data.feature_selection.k)}.</p>
        <table className="table"><thead><tr><th>Input</th><th>Score</th></tr></thead><tbody>
          {data.feature_selection.scores.map(row => <tr key={row.feature}><td>{row.feature}</td><td>{value(row.score, 4)}</td></tr>)}
        </tbody></table>
      </> : <p>Feature scores are unavailable in this saved result.</p>}
      <p className="muted">Correlation filtering, recursive feature elimination and tuned model comparisons are not included in this analysis.</p>
    </section>
  </>;
}

export default function ShotStatisticalAnalysisPage() {
  const [data, setData] = useState<ShotAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setLoading(true); setError(null); setData(null);
    fetchShotMlAnalysis().then(result => { if (active) setData(result); }).catch(reason => {
      if (active) setError(reason instanceof Error ? reason.message : "The saved analysis could not be loaded.");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [retry]);
  return <main className="container">
    <section className="card">
      <h1>Shot Statistical Analysis</h1>
      <p>Review the saved shot-data analysis. Loading this page does not train a model.</p>
      <Link href="/shot-model-metrics">Shot Model Metrics</Link>{" · "}<Link href="/shot-explorer">Shot Explorer</Link>
      <div style={{ marginTop: 16 }}><button className="btn" disabled={loading} onClick={() => setRetry(x => x + 1)}>{loading ? "Loading saved analysis…" : "Reload saved analysis"}</button></div>
    </section>
    {error && <section className="card" role="alert"><h2>Shot analysis unavailable</h2><p>{error}</p></section>}
    {data && !loading && <ShotAnalysisResults data={data} />}
  </main>;
}
