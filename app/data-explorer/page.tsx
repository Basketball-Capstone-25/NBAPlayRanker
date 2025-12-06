// This tells Next.js that this file is a *client* component
// (it runs in the browser and can use hooks like useState/useEffect).
'use client';

import { useEffect, useState } from "react";
// Import shared config + helper from our utils file:
// - API_BASE: backend URL (for the little "API call" text at the bottom)
// - SEASONS: list of seasons the data covers
// - TEAMS: list of NBA team codes
// - baselineRank: helper that calls our FastAPI /rank-plays/baseline endpoint
import { API_BASE, SEASONS, TEAMS, baselineRank } from "../utils";

// TypeScript type describing one ranked play row returned from the API.
// Each row has:
// - the play type name (e.g. "Pick and Roll Ball Handler")
// - our predicted PPP for that play
// - our offense PPP
// - opponent's allowed PPP
// - the PPP gap between us and what the opponent usually allows
type RankedPlay = {
  playType: string;
  PPP_pred: number;
  PPP_off: number;
  PPP_def: number;
  PPP_gap: number;
};

// This React component renders the "Data Explorer" page.
// It lets the user pick a season, an offensive team, and an opponent,
// then calls the baseline API and shows the Top 10 play types.
export default function DataExplorer() {
  // Dropdown state:
  // - season: which season we are looking at (default = first season in SEASONS)
  // - team: our offensive team (default TOR)
// - opp: opponent team (default BOS)
  const [season, setSeason] = useState(SEASONS[0]);
  const [team, setTeam] = useState("TOR");
  const [opp, setOpp] = useState("BOS");

  // Data + UI state:
  // - rows: the list of ranked plays from the backend
  // - loading: whether we are currently waiting for an API response
  // - error: any error message from the API call
  const [rows, setRows] = useState<RankedPlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // useEffect runs whenever [season, team, opp] change.
  // It automatically calls the baseline API and updates the table.
  useEffect(() => {
    // Safety flag to avoid setting state after the component is unmounted
    let cancelled = false;

    // This async function does the actual fetch from the backend.
    async function run() {
      try {
        setLoading(true);   // show "Loading..." message
        setError(null);     // clear previous error

        // Call our helper, which hits:
        //   /rank-plays/baseline?season=...&our=...&opp=...&k=10
        const result = await baselineRank({ season, our: team, opp, k: 10 });

        // Only update state if the effect is still active
        if (!cancelled) {
          setRows(result as RankedPlay[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          // Show a user-friendly error message if the API call fails
          setError(err?.message ?? "Unable to load baseline results.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);  // hide "Loading..." message
        }
      }
    }

    // Trigger the fetch when the effect runs
    run();

    // Cleanup function: if the component unmounts while the fetch is in flight,
    // we flip "cancelled" so we do not try to update state on an unmounted component.
    return () => {
      cancelled = true;
    };
  }, [season, team, opp]); // Dependencies: re-run whenever season/team/opp change

  // Convenience variable: the top-ranked play (first row) if it exists.
  const top = rows[0];

  // JSX: what gets rendered on the page
  return (
    <section className="card">
      <h1 className="h1">Data Explorer</h1>
      <p className="muted">
        {/* Short description explaining what this page does. */}
        This view calls the FastAPI baseline endpoint and shows the Top play-types
        for a given team and opponent.
      </p>

      {/* Filter form: season, offensive team, and opponent. */}
      <form className="form-grid" onSubmit={(e) => e.preventDefault()}>
        {/* Season dropdown */}
        <label>
          Season
          <select
            className="input"
            value={season}
            onChange={(e) => setSeason(e.target.value)}
          >
            {SEASONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        {/* Offensive team dropdown */}
        <label>
          Team (offense)
          <select
            className="input"
            value={team}
            onChange={(e) => {
              const next = e.target.value;
              setTeam(next);
              // Make sure offense and defense aren't accidentally set
              // to the same team. If they match, auto-switch opponent
              // to some other team from TEAMS.
              if (next === opp) {
                const fallback = TEAMS.find((t) => t !== next) || next;
                setOpp(fallback);
              }
            }}
          >
            {TEAMS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        {/* Defensive team dropdown */}
        <label>
          Opponent (defense)
          <select
            className="input"
            value={opp}
            onChange={(e) => setOpp(e.target.value)}
          >
            {/* We filter out the current offensive team so they can’t pick the same one. */}
            {TEAMS.filter((t) => t !== team).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </form>

      <p className="muted" style={{ marginTop: 8 }}>
        Results update automatically when you change the filters.
      </p>

      {/* Loading state */}
      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Loading baseline rankings…
        </p>
      )}

      {/* Error state */}
      {error && !loading && (
        <p
          className="muted"
          style={{ marginTop: 16, color: "#fca5a5" }}
        >
          {error}
        </p>
      )}

      {/* High-level summary of the top suggestion (only when we have data). */}
      {!loading && !error && rows.length > 0 && top && (
        <p style={{ marginTop: 16, fontSize: 13 }}>
          Top suggestion: <b>{top.playType}</b> with predicted PPP{" "}
          <b>{top.PPP_pred.toFixed(3)}</b>, which is{" "}
          {(top.PPP_gap >= 0 ? "+" : "") + top.PPP_gap.toFixed(3)} PPP
          better than this opponent usually allows.
        </p>
      )}

      {/* Main table with all Top 10 plays. */}
      {!loading && !error && rows.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>#</th>
              <th>Play type</th>
              <th>Predicted PPP</th>
              <th>Our PPP</th>
              <th>Opp allowed PPP</th>
              <th>Gap vs allowed</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              // Highlight the top row with a special CSS class "top1"
              <tr key={r.playType} className={idx === 0 ? "top1" : ""}>
                <td>{idx + 1}</td>
                <td>{r.playType}</td>
                <td>{r.PPP_pred.toFixed(3)}</td>
                <td>{r.PPP_off.toFixed(3)}</td>
                <td>{r.PPP_def.toFixed(3)}</td>
                <td>
                  {(r.PPP_gap >= 0 ? "+" : "") + r.PPP_gap.toFixed(3)} PPP
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* If everything is fine but there are no rows, show a friendly message. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No rows to show yet. Try a different team or season.
        </p>
      )}

      {/* Show the exact API call being made, so it’s clear how the frontend
          and backend connect. This is useful in the defence. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/baseline?season={season}&amp;our={team}&amp;opp={opp}&amp;k=10
        </code>
      </p>
    </section>
  );
}
