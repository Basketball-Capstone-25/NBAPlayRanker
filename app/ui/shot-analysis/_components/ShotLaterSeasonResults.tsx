import React from "react";

import type { ShotLaterSeason } from "../../../domain/shot-analysis";
const metric = (value: number | null) => value == null || !Number.isFinite(value) ? "Unavailable" : value.toFixed(4);

export function ShotLaterSeasonResults({ result, selectedModel }: { result: ShotLaterSeason; selectedModel?: string }) {
  return <section className="card">
    <h2>Later-season test: {result.test_season}</h2>
    <p>Trained on {result.train_seasons.join(", ")} ({result.train_rows.toLocaleString()} shots), then tested on {result.test_rows.toLocaleString()} shots from {result.test_season}. This season was excluded from model selection.</p>
    {selectedModel && <p>Model selected by training-season cross-validation: {selectedModel}.</p>}
    <div style={{ overflowX: "auto" }}><table className="table">
      <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
      <tbody>{result.metrics.map(row => <tr key={row.model}><td>{row.model === "training_mean" ? "Training-mean baseline" : row.model}</td><td>{metric(row.RMSE)}</td><td>{metric(row.MAE)}</td><td>{metric(row.R2)}</td></tr>)}</tbody>
    </table></div>
    <p>{result.limitation}</p>
  </section>;
}
