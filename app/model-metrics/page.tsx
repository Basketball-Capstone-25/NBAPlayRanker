// Mark this as a client component so we can use React hooks.
'use client';

import { useEffect, useState } from "react";
// fetchModelMetrics is a small helper that calls the FastAPI endpoint
// /metrics/baseline-vs-ml and returns the summary table of metrics.
import { fetchModelMetrics } from "../utils";

// TypeScript type describing one row of model metrics returned by the API.
// Each row corresponds to one model (Baseline, Ridge, RandomForest).
type MetricRow = {
  model: string;     // name of the model
  RMSE_mean: number; // mean RMSE across folds
  RMSE_std: number;  // standard deviation of RMSE across folds
  MAE_mean: number;  // mean MAE across folds
  MAE_std: number;   // standard deviation of MAE
  R2_mean: number;   // mean R² across folds
  R2_std: number;    // standard deviation of R²
};

// This component renders the "Model Performance (Baseline vs ML)" page.
// It shows the offline experiment comparing the simple statistical baseline
// and the ML models using cross-validation.
export default function ModelMetricsPage() {
  // rows: array of MetricRow that we display in the table.
  const [rows, setRows] = useState<MetricRow[]>([]);
  // nSplits: how many cross-validation folds we ask the backend to run.
  const [nSplits, setNSplits] = useState(5);
  // loading + error: UI state flags.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Whenever nSplits changes, we call the backend to re-run the comparison.
  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Call our helper which hits the FastAPI endpoint:
        //   /metrics/baseline-vs-ml?n_splits=<nSplits>
        const data = await fetchModelMetrics(nSplits);

        // The backend returns { n_splits, metrics: [...] }.
        const metrics = Array.isArray(data.metrics) ? data.metrics : [];

        if (!cancelled) {
          setRows(metrics as MetricRow[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          setError(err?.message ?? "Unable to load metrics.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup: avoid setting state after unmount.
    return () => {
      cancelled = true;
    };
  }, [nSplits]); // Re-run this effect whenever the number of folds changes.

  return (
    <section className="card">
      <h1 className="h1">Model Performance (Baseline vs ML)</h1>

      {/* Small explainer card that teaches the reader what RMSE/MAE/R² mean. */}
      <div
        className="card"
        style={{ marginTop: 8, marginBottom: 16, padding: 12 }}
      >
        <h2 className="h3">How to read these numbers</h2>
        <ul className="muted" style={{ fontSize: 13, paddingLeft: 20 }}>
          <li>
            <strong>RMSE</strong> ≈ “typical prediction error” in points per
            possession. Lower is better.
          </li>
          <li>
            <strong>MAE</strong> is another measure of error (average absolute
            difference). Also lower is better.
          </li>
          <li>
            <strong>R²</strong> tells us how much of the variation in PPP we
            explain. Closer to 1.0 is better.
          </li>
          <li>
            Every model is evaluated on the same cross-validation splits, so the
            comparison with the simple baseline is fair.
          </li>
        </ul>
      </div>

      {/* High-level description of what this page is doing. */}
      <p className="muted">
        This page shows an offline experiment on historical Synergy data. The
        baseline is a simple statistical model that uses league-average PPP for
        each season and play type. The ML models (Ridge and RandomForest) use
        a richer set of features for each team&apos;s play type.
      </p>

      {/* Control for choosing the number of cross-validation folds. */}
      <form
        className="form-grid"
        style={{ marginTop: 12, marginBottom: 12 }}
        onSubmit={(e) => e.preventDefault()}
      >
        <label>
          Cross-validation folds
          <input
            className="input"
            type="number"
            min={2}
            max={10}
            value={nSplits}
            onChange={(e) => {
              const value = Number(e.target.value) || 2;
              // Clamp to the allowed range [2, 10]
              const clamped = Math.max(2, Math.min(10, value));
              setNSplits(clamped);
            }}
          />
        </label>
      </form>

      {/* Loading state while the backend is running the comparison. */}
      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Running offline comparison…
        </p>
      )}

      {/* Error state if the API fails. */}
      {error && !loading && (
        <p className="muted" style={{ marginTop: 16, color: "#fca5a5" }}>
          {error}
        </p>
      )}

      {/* Main results table: one row per model with RMSE, MAE, R². */}
      {!loading && !error && rows.length > 0 && (
        <table className="table" style={{ marginTop: 16 }}>
          <thead>
            <tr>
              <th>Model</th>
              <th>RMSE (↓)</th>
              <th>MAE (↓)</th>
              <th>R² (↑)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model}>
                <td>{r.model}</td>
                <td>
                  {r.RMSE_mean.toFixed(4)}{" "}
                  <span className="muted">
                    (±{r.RMSE_std.toFixed(4)})
                  </span>
                </td>
                <td>
                  {r.MAE_mean.toFixed(4)}{" "}
                  <span className="muted">
                    (±{r.MAE_std.toFixed(4)})
                  </span>
                </td>
                <td>
                  {r.R2_mean.toFixed(4)}{" "}
                  <span className="muted">
                    (±{r.R2_std.toFixed(4)})
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Friendly message if there’s simply no data yet. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No results yet. Try a different number of folds.
        </p>
      )}

      {/* Note about how this page fits into the product roadmap. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        This endpoint is for analysis and defence only. In a future phase, the
        best-performing ML model would be persisted and exposed via a separate
        live ranking endpoint.
      </p>
    </section>
  );
}
