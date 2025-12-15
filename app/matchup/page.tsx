// app/matchup-console/page.tsx
//
// This file implements the **Matchup Console (Baseline)** screen.
//
// Purpose (for defence):
// - Show a clean, worked example of our **baseline recommendation engine**
//   for a single matchup.
// - Use a fixed Season + Our team + Opponent so we can talk through the logic
//   without changing inputs during the demo.
// - Display the full Top-K ranking returned by the baseline API, including:
//     * predicted PPP for each play type,
//     * our team’s PPP on that play type,
//     * the opponent’s allowed PPP on that play type,
//     * and the gap vs the opponent’s typical allowed PPP.
// - Make the exact API call visible so the frontend–backend contract is clear.
//
// This page is intentionally focused on a single example matchup.
// If the user wants to change matchups, they do that on the **Data Explorer**
// page, which is dedicated to exploring and exporting inputs.

// Mark this as a client component because we use React hooks.
'use client';

import { useEffect, useState } from "react";

// Shared utilities:
// - API_BASE: base URL for our FastAPI backend.
// - SEASONS: list of seasons from the Synergy snapshot.
// - baselineRank: helper that calls `/rank-plays/baseline` with query params
//   and returns parsed JSON.
//
// We import these from a central utils module so the page stays focused on UI
// and presentation logic.
import { API_BASE, SEASONS, baselineRank } from "../utils";

// Shape of each ranked play returned by the baseline API.
//
// The backend computes these fields from team-level tables and league averages:
// - playType: label like "Spotup", "PRRollMan", etc.
// - PPP_pred: predicted PPP for this play type in the selected matchup.
// - PPP_off: our team's historical PPP for that play type (after shrinkage).
// - PPP_def: opponent's historical PPP allowed on that play type (after shrinkage).
// - PPP_gap: PPP_pred − PPP_def (how much better/worse than their norm).
//
// Modelling these explicitly gives us type-safety and makes it clear what the
// baseline model is actually producing.
type RankedPlay = {
  playType: string;
  PPP_pred: number;
  PPP_off: number;
  PPP_def: number;
  PPP_gap: number;
};

// A single, fixed example request used on this screen.
//
// Rationale:
// - We want a **stable example** for the defence, so the numbers don't change
//   mid-presentation.
// - Using a constant object also makes it easy to show the exact API call
//   at the bottom of the page.
//
// Season: we use the first entry in SEASONS (e.g., "2019-20").
// Teams: Toronto (offense) vs Boston (defense).
// k: we request the Top-10 play types from the baseline model.
const EXAMPLE_REQUEST = {
  season: SEASONS[0], // e.g., "2019-20"
  our: "TOR",
  opp: "BOS",
  k: 10,
};

// Main React component for the Matchup Console page.
export default function MatchupConsolePage() {
  // --------------------------
  // 1. State: data + UI flags
  // --------------------------

  // rows: array of RankedPlay objects returned by the baseline API.
  const [rows, setRows] = useState<RankedPlay[]>([]);

  // loading: true while we're waiting for the backend to respond.
  const [loading, setLoading] = useState(false);

  // error: holds any error message from the fetch, or null if none.
  const [error, setError] = useState<string | null>(null);

  // -------------------------------------------------------
  // 2. Effect: call the baseline endpoint once on mount
  // -------------------------------------------------------
  //
  // When the component first mounts, we:
  //   1. mark loading=true,
  //   2. call baselineRank(EXAMPLE_REQUEST),
  //   3. store the resulting rows,
  //   4. handle any errors,
  //   5. mark loading=false.
  //
  // We don't re-run this effect because the dependency array is []:
  // this page is designed around a **single example matchup**.

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Call the shared helper which issues a GET request to:
        //   /rank-plays/baseline?season=...&our=...&opp=...&k=...
        // and returns parsed JSON.
        const result = await baselineRank(EXAMPLE_REQUEST);

        if (!cancelled) {
          // The backend should return an array of objects. We narrow to RankedPlay[].
          setRows(Array.isArray(result) ? (result as RankedPlay[]) : []);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          // Capture a friendly error message and clear any stale rows.
          setError(err?.message ?? "Unable to load baseline ranking.");
          setRows([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup: if the component unmounts before the request finishes,
    // mark it as cancelled so we don't try to set state on an unmounted component.
    return () => {
      cancelled = true;
    };
  }, []);

  // Convenience: the top-ranked play, if present.
  const top = rows[0];

  // -------------------------------------------------------
  // 3. Render: JSX for the Matchup Console page
  // -------------------------------------------------------

  return (
    <section className="card">
      {/* Title and short description of the use-case. */}
      <h1 className="h1">Matchup Console (Baseline)</h1>
      <p className="muted">
        This page shows a worked example of our baseline recommendation engine for a
        single matchup. The example uses a fixed season and team vs opponent. If you want
        to change the matchup, you do that on the <strong>Data Explorer</strong> page and
        use the CSV output there.
      </p>

      {/* ------------------------------ */}
      {/* High-level explanation of logic */}
      {/* ------------------------------ */}
      <div
        className="card"
        style={{ marginTop: 8, marginBottom: 16, padding: 12 }}
      >
        <h2 className="h3">What the baseline is doing</h2>
        <ul className="muted" style={{ fontSize: 13, paddingLeft: 20 }}>
          <li>
            Take our team’s historical PPP by play type from the Synergy snapshot.
          </li>
          <li>
            Take the opponent’s historical PPP allowed by play type from the same data.
          </li>
          <li>
            Combine them into a predicted PPP for each play type (shrinkage toward
            league averages to avoid overfitting small samples).
          </li>
          <li>Sort play types from highest to lowest predicted PPP and keep Top-K.</li>
        </ul>
      </div>

      {/* ------------------------------ */}
      {/* Loading / error / empty states */}
      {/* ------------------------------ */}

      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Loading baseline ranking…
        </p>
      )}

      {error && !loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          {error}
        </p>
      )}

      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No recommendations returned for this example matchup yet.
        </p>
      )}

      {/* ------------------------------------------------- */}
      {/* Optional small summary for the top recommended play */}
      {/* ------------------------------------------------- */}
      {!loading && !error && top && (
        <p className="muted" style={{ marginTop: 16 }}>
          For this example matchup, the top recommended play type is{" "}
          <b>{top.playType}</b> with predicted PPP <b>{top.PPP_pred.toFixed(3)}</b>. That
          is{" "}
          {(top.PPP_gap >= 0 ? "+" : "") + top.PPP_gap.toFixed(3)} PPP better than what
          the opponent usually allows on that play.
        </p>
      )}

      {/* ------------------------------ */}
      {/* Main Top-K ranking table        */}
      {/* ------------------------------ */}
      {!loading && !error && rows.length > 0 && (
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table className="table">
            <thead>
              <tr>
                <th>#</th>
                <th>Play type</th>
                <th>Predicted PPP</th>
                <th>Our PPP</th>
                <th>Opponent allowed PPP</th>
                <th>Gap vs allowed</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => (
                // We add a CSS class "top1" to the first row so it can be
                // highlighted via styling, making the primary recommendation
                // immediately visible to the coach.
                <tr key={r.playType} className={idx === 0 ? "top1" : ""}>
                  <td>{idx + 1}</td>
                  <td>{r.playType}</td>
                  <td>{r.PPP_pred.toFixed(3)}</td>
                  <td>{r.PPP_off.toFixed(3)}</td>
                  <td>{r.PPP_def.toFixed(3)}</td>
                  <td>{(r.PPP_gap >= 0 ? "+" : "") + r.PPP_gap.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ------------------------------ */}
      {/* API call trace for defence      */}
      {/* ------------------------------ */}
      {/* For explanation and debugging, we show the exact baseline endpoint
          and query parameters used for this example. This makes the flow
          from frontend → backend → model completely transparent. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/baseline?season={EXAMPLE_REQUEST.season}&amp;our=
          {EXAMPLE_REQUEST.our}
          &amp;opp={EXAMPLE_REQUEST.opp}&amp;k={EXAMPLE_REQUEST.k}
        </code>
      </p>
    </section>
  );
}
