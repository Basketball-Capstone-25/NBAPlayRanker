import React from "react";

import type { ShotLaterSeason } from "../../../domain/shot-analysis";
const metric = (value: number | null) => value == null || !Number.isFinite(value) ? "Unavailable" : value.toFixed(4);

export const shotModelName = (name: string) => ({ gbr: "Gradient boosting", ridge: "Ridge regression", rf: "Random forest", training_mean: "Training-mean baseline" }[name] || name);

export function ShotLaterSeasonResults({ result, selectedModel }: { result: ShotLaterSeason; selectedModel?: string }) {
  return <section className="card">
    <h2>Later-season test: {result.test_season}</h2>
    <p>Tested on {result.test_rows.toLocaleString()} shots from {result.test_season}, a season excluded from model selection.</p>
    {selectedModel && <p>Selected model: {shotModelName(selectedModel)}.</p>}
    <div style={{ overflowX: "auto" }}><table className="table">
      <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
      <tbody>{result.metrics.map(row => <tr key={row.model}><td>{shotModelName(row.model)}</td><td>{metric(row.RMSE)}</td><td>{metric(row.MAE)}</td><td>{metric(row.R2)}</td></tr>)}</tbody>
    </table></div>
    <p>{result.limitation}</p>
    <details><summary>Training sample</summary><p>Trained on {result.train_rows.toLocaleString()} shots from {result.train_seasons.join(", ")}.</p></details>
  </section>;
}
