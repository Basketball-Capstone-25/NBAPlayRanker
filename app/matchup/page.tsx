'use client';

import { useEffect, useState } from "react";
import { API_BASE, SEASONS, baselineRank } from "../utils";

// Shape of each ranked play returned by the baseline API.
type RankedPlay = {
  playType: string;
  PPP_pred: number;
  PPP_off: number;
  PPP_def: number;
  PPP_gap: number;
};

// A single example request used on this screen.
// You can mention these defaults verbally when you present.
const EXAMPLE_REQUEST = {
  season: SEASONS[0], // e.g., "2019-20"
  our: "TOR",
  opp: "BOS",
  k: 10,
};

export default function MatchupConsolePage() {
  const [rows, setRows] = useState<RankedPlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Call the baseline endpoint once on mount for the example matchup.
  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        const result = await baselineRank(EXAMPLE_REQUEST);

        if (!cancelled) {
          setRows(Array.isArray(result) ? (result as RankedPlay[]) : []);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
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

    return () => {
      cancelled = true;
    };
  }, []);

  const top = rows[0];

  return (
    <section className="card">
      <h1 className="h1">Matchup Console (Baseline)</h1>
      <p className="muted">
        This page shows a worked example of our baseline recommendation engine for a
        single matchup. The example uses a fixed season and team vs opponent. If you want
        to change the matchup, you do that on the <strong>Data Explorer</strong> page and
        use the CSV output there.
      </p>

      {/* High-level explanation of the logic */}
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

      {/* Loading / error / empty states */}
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

      {/* Optional small summary for the top play */}
      {!loading && !error && top && (
        <p className="muted" style={{ marginTop: 16 }}>
          For this example matchup, the top recommended play type is{" "}
          <b>{top.playType}</b> with predicted PPP <b>{top.PPP_pred.toFixed(3)}</b>. That
          is{" "}
          {(top.PPP_gap >= 0 ? "+" : "") + top.PPP_gap.toFixed(3)} PPP better than what
          the opponent usually allows on that play.
        </p>
      )}

      {/* Main Top-K table */}
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

      {/* API call for defence explanation */}
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
