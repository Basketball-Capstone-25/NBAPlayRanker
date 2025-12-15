// app/data-explorer/page.tsx
//
// This file implements the Data Explorer screen.
//
// Purpose
// - Let the user choose Season + Offense team + Opponent.
// - Call the baseline ranking endpoint and preview the **underlying PPP inputs**
//   for that matchup (our offense PPP and opponent defense PPP allowed).
// - Provide a CSV export link so analysts can download the same table and do
//   deeper offline analysis in Excel / R / Python.
// - Make the exact API call visible, so the contract between frontend and
//   FastAPI is clear.
//
// This page is intentionally **about the inputs** to the baseline model, not the
// full predicted PPP and gap columns. The prediction and ranking logic are
// demonstrated more explicitly on the Matchup Console page.

// Mark this as a client component so we can use React hooks.
'use client';

import { useEffect, useState } from "react";

// Shared utilities:
//
// - API_BASE: base URL for our FastAPI backend.
// - SEASONS: list of seasons available in the Synergy snapshot.
// - TEAMS: list of team codes.
// - baselineRank: helper that calls `/rank-plays/baseline` with query params
//   and returns the parsed JSON response.
//
// Centralising these in ../utils keeps this component focused on UI logic.
import { API_BASE, SEASONS, TEAMS, baselineRank } from "../utils";

// Minimal subset of the backend response we care about on this screen.
//
// The backend returns richer objects (including predicted PPP and PPP gap), but
// here we only need:
// - playType: label for the row (e.g., "Spotup").
// - PPP_off: our team's offensive PPP for that play type.
// - PPP_def: opponent's defensive PPP allowed for that play type.
//
// Restricting the type like this makes it clear that this page is showing
// baseline **inputs**, not the full recommendation output.
type RankedPlay = {
  playType: string;
  PPP_off: number;
  PPP_def: number;
};

// Main React component for the Data Explorer page.
export default function DataExplorerPage() {
  // --------------------------
  // 1. Filter state
  // --------------------------

  // Season selector:
  // - Default to the first season in SEASONS.
  //   (You could change this to the most recent; we use index 0 to keep it simple.)
  const [season, setSeason] = useState(SEASONS[0]);

  // Offense team ("our" team).
  // - Default to TOR to stay consistent with other examples in the app.
  const [team, setTeam] = useState("TOR");

  // Opponent team (defense).
  // - Default to BOS to match the example used on the Matchup Console.
  const [opp, setOpp] = useState("BOS");

  // --------------------------
  // 2. Data + UI state
  // --------------------------

  // Rows returned from the baseline endpoint (trimmed to RankedPlay shape).
  const [rows, setRows] = useState<RankedPlay[]>([]);

  // Loading + error flags for user feedback.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // -------------------------------------------------------
  // 3. Effect: fetch data whenever filters change
  // -------------------------------------------------------
  //
  // Whenever the Season, Offense team, or Opponent change, we:
  //   1. set loading=true,
  //   2. call baselineRank(...) with k=10,
  //   3. store the resulting rows (or error),
  //   4. set loading=false.
  //
  // We treat this as a lightweight "preview" of the baseline input table:
  // - Top 10 rows is enough to show the structure and values without overloading
  //   the UI.
  // - Analysts can click "Download CSV" to work with the full data offline.

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Call the shared helper for the baseline ranking endpoint.
        //
        // Parameters:
        // - season: selected season.
        // - our: offense team.
        // - opp: opponent team.
        // - k: how many rows to request.
        //
        // IMPORTANT: k MUST be <= 10 to satisfy FastAPI validation. We use 10
        // here as a small preview in the UI.
        const result = await baselineRank({
          season,
          our: team,
          opp,
          k: 10,
        });

        if (!cancelled) {
          // The backend returns an array of objects. We narrow it to RankedPlay.
          // If the response is not an array, we treat it as empty.
          setRows(Array.isArray(result) ? (result as RankedPlay[]) : []);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          // Show a friendly error message and clear out any stale rows.
          setError(err?.message ?? "Unable to load data.");
          setRows([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup: mark the request as cancelled if the component unmounts.
    return () => {
      cancelled = true;
    };
  }, [season, team, opp]);
  // Dependency array:
  // - When any of these filter values change, we re-fetch.

  // -------------------------------------------------------
  // 4. Derived: CSV export URL for the current filters
  // -------------------------------------------------------
  //
  // This uses a `.csv` variant of the same baseline endpoint so analysts can
  // download a file and do more detailed analysis outside the app.
  //
  // We use encodeURIComponent on the query values to be safe.
  const csvUrl = `${API_BASE}/rank-plays/baseline.csv?season=${encodeURIComponent(
    season
  )}&our=${encodeURIComponent(team)}&opp=${encodeURIComponent(opp)}&k=10`;

  // -------------------------------------------------------
  // 5. Render: JSX for the Data Explorer page
  // -------------------------------------------------------

  return (
    <section className="card">
      {/* Title + explanation at the top. */}
      <h1 className="h1">Data Explorer</h1>
      <p className="muted">
        Preview the underlying team–season–play-type matchup data that feeds the baseline
        model. Analysts can export the current view as CSV and do deeper work in Excel, R,
        or Python.
      </p>

      {/* -------------------- */}
      {/* Filter form          */}
      {/* -------------------- */}
      {/* We wrap the selects in a form element purely for grouping and styling.
          onSubmit is prevented because all changes are handled via onChange. */}
      <form className="form-grid" onSubmit={(e) => e.preventDefault()}>
        {/* Season selector:
            - Lets the user decide which Synergy snapshot to query. */}
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

        {/* Offense team selector ("our team"):
            - Updates the "team" state.
            - If the user accidentally picks the same team as the opponent,
              we automatically change the opponent to the first different team
              so that we never send our=opp to the backend. */}
        <label>
          Offense (our team)
          <select
            className="input"
            value={team}
            onChange={(e) => {
              const next = e.target.value;
              setTeam(next);

              // Guard against our=opp by auto-adjusting the opponent.
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

        {/* Opponent selector:
            - Updates the "opp" state.
            - If the user picks the same team as "our" team, we automatically
              change the offense team to a different fallback value. */}
        <label>
          Opponent
          <select
            className="input"
            value={opp}
            onChange={(e) => {
              const next = e.target.value;
              setOpp(next);

              // Guard in the other direction: if opponent == offense, adjust offense.
              if (next === team) {
                const fallback = TEAMS.find((t) => t !== next) || next;
                setTeam(fallback);
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
      </form>

      {/* -------------------- */}
      {/* Context bar + CSV    */}
      {/* -------------------- */}
      <div
        style={{
          marginTop: 16,
          marginBottom: 8,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        {/* Text that summarises what the table is showing. */}
        <p className="muted" style={{ fontSize: 12 }}>
          Showing baseline input stats for <b>{team}</b> vs <b>{opp}</b> in{" "}
          <b>{season}</b>.
        </p>

        {/* CSV export link:
            - Points directly at the csvUrl we constructed above.
            - Opens in a new tab so the app stays open. */}
        <a
          href={csvUrl}
          className="btn"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download CSV
        </a>
      </div>

      {/* -------------------- */}
      {/* Loading / error states */}
      {/* -------------------- */}

      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Loading data…
        </p>
      )}

      {error && !loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          {error}
        </p>
      )}

      {/* Case when the request succeeded but returned no rows. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No rows returned for this combination yet.
        </p>
      )}

      {/* -------------------- */}
      {/* Main data table       */}
      {/* -------------------- */}
      {/* Only render the table when we have rows and no error. */}
      {!loading && !error && rows.length > 0 && (
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table className="table">
            <thead>
              <tr>
                <th>#</th>
                <th>Play type</th>
                <th>Offense PPP (our team)</th>
                <th>Defense PPP allowed (opponent)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => (
                <tr key={r.playType}>
                  {/* Simple 1-based index for row numbering. */}
                  <td>{idx + 1}</td>
                  <td>{r.playType}</td>
                  {/* Use toFixed(3) for consistent decimal formatting. */}
                  <td>{r.PPP_off?.toFixed(3)}</td>
                  <td>{r.PPP_def?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* -------------------- */}
      {/* API call trace        */}
      {/* -------------------- */}
      {/* For defence and debugging, we show the exact URL that the baselineRank
          helper constructs. This makes the frontend–backend contract explicit. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/baseline?season={season}&amp;our={team}
          &amp;opp={opp}&amp;k=10
        </code>
      </p>
    </section>
  );
}
