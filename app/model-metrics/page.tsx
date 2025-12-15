// app/model-performance/page.tsx
//
// This file implements the **Model Performance (Baseline vs ML)** screen.
//
// Purpose (for defence):
// - Show an **offline experiment** comparing our simple statistical baseline
//   model to machine learning models (Ridge and RandomForest).
// - Use cross-validation on historical Synergy data to estimate how well each
//   model predicts PPP (points per possession) for team–play-type rows.
// - Surface three standard regression metrics:
//     * RMSE  = typical prediction error (lower is better)
//     * MAE   = average absolute error (lower is better)
//     * R²    = proportion of variance explained (closer to 1 is better)
// - Allow the user to change the number of cross-validation folds (n_splits)
//   and re-run the comparison through the backend.
// - Make it explicit that this is an analysis endpoint only; the “winner”
//   model would be deployed as a separate live ranking endpoint in later phases.

// Mark this as a client component so we can use React hooks.
'use client';

import { useEffect, useState } from "react";

// fetchModelMetrics is a helper exported from ../utils.
//
// It hides the low-level fetch call to FastAPI. Under the hood it:
//   - builds the URL `/metrics/baseline-vs-ml?n_splits=<nSplits>`,
//   - performs a GET request,
//   - parses the JSON,
//   - and returns an object like: { n_splits: number, metrics: MetricRow[] }.
//
// Keeping this in a shared utils module keeps this component focused on UI logic.
import { fetchModelMetrics } from "../utils";

// TypeScript type describing one row of model metrics returned by the API.
//
// Each row corresponds to one model:
//   - "Baseline (league mean)"
//   - "RandomForest"
//   - "Ridge"
// and contains summary stats of its performance across all cross-validation folds.
type MetricRow = {
  model: string;     // human-readable name of the model
  RMSE_mean: number; // mean RMSE across folds
  RMSE_std: number;  // standard deviation of RMSE across folds
  MAE_mean: number;  // mean MAE across folds
  MAE_std: number;   // standard deviation of MAE across folds
  R2_mean: number;   // mean R² across folds
  R2_std: number;    // standard deviation of R²
};

// Main React component for the Model Performance page.
//
// This page is **read-only analytics**: it never mutates the training data,
// it just requests metrics that the backend computes on demand.
export default function ModelMetricsPage() {
  // --------------------------
  // 1. State: data + UI flags
  // --------------------------

  // rows: array of MetricRow objects to display in the table.
  const [rows, setRows] = useState<MetricRow[]>([]);

  // nSplits: number of cross-validation folds to ask the backend to run.
  //
  // Cross-validation idea (for defence):
  // - Split the historical dataset into `nSplits` folds.
  // - Repeatedly train on (nSplits - 1) folds and evaluate on the held-out fold.
  // - Aggregate the metrics (mean ± std) across all folds.
  //
  // We initialise this to 5, which is a common “balanced” choice.
  const [nSplits, setNSplits] = useState(5);

  // Loading + error: standard async UI flags.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // -------------------------------------------------------
  // 2. Effect: re-fetch metrics whenever nSplits changes
  // -------------------------------------------------------
  //
  // When the component mounts, and whenever the user changes the
  // "Cross-validation folds" input, we:
  //   1. set loading=true and clear any previous error,
  //   2. call fetchModelMetrics(nSplits),
  //   3. extract the metrics array from the response,
  //   4. store it in `rows`,
  //   5. handle any errors,
  //   6. set loading=false.
  //
  // We also guard against setting state after unmount using a `cancelled` flag.

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Call our helper which hits the FastAPI endpoint:
        //   GET /metrics/baseline-vs-ml?n_splits=<nSplits>
        //
        // Backend behaviour (for defence explanation):
        // - For each model (Baseline, Ridge, RandomForest):
        //     * run K-fold cross-validation with K = n_splits
        //     * compute RMSE, MAE, R² on each fold
        //     * aggregate mean and standard deviation
        // - Return an object with n_splits and metrics[].
        const data = await fetchModelMetrics(nSplits);

        // The backend returns { n_splits, metrics: [...] }.
        // We defensive-check that metrics is an array before casting.
        const metrics = Array.isArray(data.metrics) ? data.metrics : [];

        if (!cancelled) {
          setRows(metrics as MetricRow[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          // Surface a friendly error message; we don't clear rows here so
          // the user can still see the last successful result if they want.
          setError(err?.message ?? "Unable to load metrics.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup: if the component unmounts before the request finishes,
    // this prevents React state updates on an unmounted component.
    return () => {
      cancelled = true;
    };
  }, [nSplits]); // Re-run this effect whenever the number of folds changes.

  // -------------------------------------------------------
  // 3. Render: JSX for the Model Performance page
  // -------------------------------------------------------

  return (
    <section className="card">
      <h1 className="h1">Model Performance (Baseline vs ML)</h1>

      {/* --------------------------------------------- */}
      {/* Explainer card: what RMSE / MAE / R² actually mean */}
      {/* --------------------------------------------- */}
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

      {/* High-level description of what this experiment is doing. */}
      <p className="muted">
        This page shows an offline experiment on historical Synergy data. The
        baseline is a simple statistical model that uses league-average PPP for
        each season and play type. The ML models (Ridge and RandomForest) use
        a richer set of features for each team&apos;s play type.
      </p>

      {/* --------------------------------------------- */}
      {/* Control for choosing the number of CV folds    */}
      {/* --------------------------------------------- */}
      {/* We wrap the input in a small form for layout; submission is prevented
          because everything happens on change. */}
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
              // Clamp to the allowed range [2, 10] so the backend never gets
              // an invalid n_splits value.
              const clamped = Math.max(2, Math.min(10, value));
              setNSplits(clamped);
            }}
          />
        </label>
      </form>

      {/* ------------------------------ */}
      {/* Loading + error states          */}
      {/* ------------------------------ */}

      {/* While the backend is running the cross-validation, we show a simple
          text indicator rather than a stale table. */}
      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Running offline comparison…
        </p>
      )}

      {/* If the API call failed, show the error in a soft red color. */}
      {error && !loading && (
        <p className="muted" style={{ marginTop: 16, color: "#fca5a5" }}>
          {error}
        </p>
      )}

      {/* --------------------------------------------- */}
      {/* Main results table: one row per model          */}
      {/* --------------------------------------------- */}
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

      {/* Friendly message if there’s simply no data for the current settings. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No results yet. Try a different number of folds.
        </p>
      )}

      {/* --------------------------------------------- */}
      {/* Note about how this fits into the roadmap      */}
      {/* --------------------------------------------- */}
      {/* This text makes clear that this endpoint is analytical and that
          deployment of the best model is a deliberate later step. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        This endpoint is for analysis and defence only. In a future phase, the
        best-performing ML model would be persisted and exposed via a separate
        live ranking endpoint.
      </p>
    </section>
  );
}
