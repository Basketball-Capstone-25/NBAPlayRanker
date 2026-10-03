"use client";

import { useEffect, useState } from "react";
import { fetchCalibration, type CalibrationReport } from "../../../services/calibration";

const format = (value: number | null) => value === null ? "—"
  : value.toFixed(value !== 0 && Math.abs(value) < 0.0001 ? 6 : 4);

export default function CalibrationPanel({ nSplits }: { nSplits: number }) {
  const [report, setReport] = useState<CalibrationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setReport(null);
    setError(null);
    fetchCalibration(nSplits, controller.signal).then(result => {
      if (!controller.signal.aborted) setReport(result);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Unable to load calibration.");
    });
    return () => controller.abort();
  }, [nSplits, attempt]);

  const points = report?.summary.bins.filter(bin => bin.count > 0) ?? [];
  const values = points.flatMap(bin => [bin.mean_predicted_ppp!, bin.mean_realized_ppp!]);
  const minimum = values.length ? Math.floor(Math.min(...values) * 10) / 10 : 0;
  const maximum = values.length ? Math.max(minimum + 0.1, Math.ceil(Math.max(...values) * 10) / 10) : 1;
  const x = (value: number) => 64 + (value - minimum) / (maximum - minimum) * 430;
  const y = (value: number) => 316 - (value - minimum) / (maximum - minimum) * 270;

  return <section aria-labelledby="calibration-heading">
    <h2 id="calibration-heading">Predicted versus realized PPP</h2>
    <p className="muted">Fixed Ridge model, tested on later seasons. Positive bias means PPP is overpredicted;
      negative bias means it is underpredicted. This check does not change the model.</p>
    {error ? <div role="alert"><p>{error}</p><button type="button" className="btn" onClick={() => setAttempt(value => value + 1)}>Retry calibration</button></div>
      : !report ? <p role="status">Computing held-out calibration…</p> : <>
        <div className="grid">
          {([
            ["Signed bias", report.summary.bias_ppp],
            ["MAE", report.summary.mae_ppp],
            ["RMSE", report.summary.rmse_ppp],
          ] as const).map(([label, value]) => <div className="kpi" key={label}>
            <div className="label">{label}</div><strong>{format(value)} PPP</strong>
          </div>)}
        </div>
        <p>{report.summary.count.toLocaleString()} held-out team/play-type rows across {report.evaluation.n_splits} seasons.
          Mean predicted: {format(report.summary.mean_predicted_ppp)} PPP; realized: {format(report.summary.mean_realized_ppp)} PPP.</p>
        <svg viewBox="0 0 560 370" role="img" aria-labelledby="calibration-chart-title calibration-chart-description" style={{ width: "100%", maxWidth: 620 }}>
          <title id="calibration-chart-title">Mean predicted versus realized PPP by prediction bin</title>
          <desc id="calibration-chart-description">Each point is a populated prediction bin. Points below the diagonal overpredict PPP. The table below contains exact values.</desc>
          <line x1="64" y1="316" x2="494" y2="316" stroke="currentColor" />
          <line x1="64" y1="316" x2="64" y2="46" stroke="currentColor" />
          <line x1="64" y1="316" x2="494" y2="46" stroke="#64748b" strokeDasharray="5 5" />
          <text x="64" y="336" fontSize="12">{minimum.toFixed(1)}</text><text x="474" y="336" fontSize="12">{maximum.toFixed(1)}</text>
          <text x="29" y="319" fontSize="12">{minimum.toFixed(1)}</text><text x="29" y="50" fontSize="12">{maximum.toFixed(1)}</text>
          <text x="220" y="360" fontSize="13">Mean predicted PPP</text>
          <text x="18" y="245" fontSize="13" transform="rotate(-90 18 245)">Mean realized PPP</text>
          {points.map((bin, index) => <circle key={index} cx={x(bin.mean_predicted_ppp!)} cy={y(bin.mean_realized_ppp!)} r={bin.sparse ? 4 : 6} fill={bin.sparse ? "#92400e" : "#2563eb"}>
            <title>{bin.count} rows; predicted {format(bin.mean_predicted_ppp)}, realized {format(bin.mean_realized_ppp)}, bias {format(bin.bias_ppp)} PPP{bin.sparse ? "; sparse bin" : ""}</title>
          </circle>)}
        </svg>
        <div style={{ overflowX: "auto" }}><table className="table">
          <caption>Equal-width prediction bins; sparse means fewer than 20 rows. Empty bins have no mean.</caption>
          <thead><tr><th scope="col">Predicted range</th><th scope="col">Rows</th><th scope="col">Predicted PPP</th><th scope="col">Realized PPP</th><th scope="col">Bias (PPP)</th></tr></thead>
          <tbody>{report.summary.bins.map((bin, index) => <tr key={index}>
            <td>{format(bin.lower_ppp)}–{format(bin.upper_ppp)}</td><td>{bin.count}{bin.sparse ? " (sparse)" : ""}</td>
            <td>{format(bin.mean_predicted_ppp)}</td><td>{format(bin.mean_realized_ppp)}</td><td>{format(bin.bias_ppp)}</td>
          </tr>)}</tbody>
        </table></div>
        <h3>Evaluation provenance</h3>
        <p>{report.evaluation.method}; Ridge alpha {report.evaluation.model_parameters.alpha}.
          Scaling uses {report.evaluation.scaler_fit}. {report.evaluation.weighting}.</p>
        <div style={{ overflowX: "auto" }}><table className="table">
          <caption>Each held-out season is strictly later than its training seasons.</caption>
          <thead><tr><th scope="col">Training seasons</th><th scope="col">Test season</th><th scope="col">Test rows</th><th scope="col">Bias (PPP)</th><th scope="col">RMSE (PPP)</th></tr></thead>
          <tbody>{report.folds.map(fold => <tr key={fold.test_season}>
            <td>{fold.train_seasons.join(", ")}</td><td>{fold.test_season}</td><td>{fold.count}</td><td>{format(fold.bias_ppp)}</td><td>{format(fold.rmse_ppp)}</td>
          </tr>)}</tbody>
        </table></div>
        <ul className="muted">{report.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>
      </>}
  </section>;
}
