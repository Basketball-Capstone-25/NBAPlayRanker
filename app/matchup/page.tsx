// Mark this as a client component so we can use React hooks in Next.js.
'use client';

import { useEffect, useState } from "react";
// Shared utilities:
// - API_BASE: backend URL for FastAPI
// - SEASONS: list of seasons
// - TEAMS: list of team abbreviations
// - baselineRank: helper that calls /rank-plays/baseline on the backend
import { API_BASE, SEASONS, TEAMS, baselineRank } from "../utils";

// TypeScript type describing one ranked play returned by the baseline API.
type RankedPlay = {
  playType: string;  // name of the play type (e.g. "Pick and Roll Ball Handler")
  PPP_pred: number;  // predicted points per possession for this matchup
  PPP_off: number;   // our team's shrunk PPP for this play type
  PPP_def: number;   // opponent's shrunk PPP allowed for this play type
  PPP_gap: number;   // difference between our PPP and what the opponent usually allows
};

// This component renders the "Matchup Console" page.
// It is a coach-facing view for a single game matchup.
export default function Matchup() {
  // Dropdown state:
  // which season, which is our team, which is the opponent, and how many plays (K).
  const [season, setSeason] = useState(SEASONS[0]);
  const [our, setOur] = useState("TOR");
  const [opp, setOpp] = useState("BOS");
  const [k, setK] = useState(5);

  // Data + UI state:
  // - rows: list of ranked play types returned from the baseline model
  // - loading: whether we're waiting for the API
  // - error: any error message if the API call fails
  const [rows, setRows] = useState<RankedPlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // When the coach clicks "Download CSV", we build the same query parameters
  // and open the CSV endpoint in a new tab. This lets them pull the Top-K
  // list into Excel or a scouting document.
  const handleDownloadCsv = () => {
    const params = new URLSearchParams({
      season,
      our,
      opp,
      k: String(k),
    });
    const url = `${API_BASE}/rank-plays/baseline.csv?${params.toString()}`;
    window.open(url, "_blank");
  };

  // useEffect calls the baseline API whenever season / our / opp / k change.
  useEffect(() => {
    let cancelled = false; // safety flag for cleanup

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Call the baseline model via our helper; this hits FastAPI:
        //   /rank-plays/baseline?season=...&our=...&opp=...&k=...
        const result = await baselineRank({ season, our, opp, k });

        if (!cancelled) {
          setRows(result as RankedPlay[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          setError(err?.message ?? "Unable to load matchup results.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup: if the component unmounts while the request is in-flight,
    // mark it as cancelled so we don't try to update unmounted state.
    return () => {
      cancelled = true;
    };
  }, [season, our, opp, k]); // re-run the effect whenever any of these change

  // Convenience pointer to the best-ranked play (first row) if it exists.
  const top = rows[0];

  // JSX: layout for the coach-facing matchup console
  return (
    <section className="card">
      <h1 className="h1">Matchup Console (Baseline)</h1>
      <p className="muted">
        {/* High-level explanation of this page for users and for your defence. */}
        Coach view for a single game. Choose a season, your team, an opponent, and how
        many plays (K) you want in the ranked list. The UI calls the FastAPI baseline
        endpoint and shows a sorted Top-K list.
      </p>

      {/* Filter form: which season, which teams, and how many Top-K plays. */}
      <form className="form-grid" onSubmit={(e) => e.preventDefault()}>
        {/* Season selector */}
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

        {/* Our team selector (offense) */}
        <label>
          Our team
          <select
            className="input"
            value={our}
            onChange={(e) => setOur(e.target.value)}
          >
            {TEAMS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        {/* Opponent team selector (defense) */}
        <label>
          Opponent
          <select
            className="input"
            value={opp}
            onChange={(e) => setOpp(e.target.value)}
          >
            {TEAMS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        {/* Slider / numeric input for K: how many plays to show. */}
        <label>
          Top-K plays
          <input
            className="input"
            type="number"
            min={1}
            max={10}
            value={k}
            onChange={(e) => {
              const value = Number(e.target.value) || 1;
              // Clamp between 1 and 10 so we don't ask the backend for silly values.
              const clamped = Math.min(10, Math.max(1, value));
              setK(clamped);
            }}
          />
        </label>
      </form>

      {/* Button to export the current Top-K ranking as CSV for coaches. */}
      <div
        style={{
          marginTop: 8,
          display: "flex",
          gap: 8,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <button
          type="button"
          className="button"
          onClick={handleDownloadCsv}
        >
          Download Top {k} as CSV
        </button>
        <span className="muted" style={{ fontSize: 11 }}>
          Opens a CSV version of this Top-K list for use in Excel / scouting docs.
        </span>
      </div>

      {/* Loading state: message while the baseline model is running. */}
      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Running baseline model…
        </p>
      )}

      {/* Error state: show error message if the API failed. */}
      {error && !loading && (
        <p
          className="muted"
          style={{ marginTop: 16, color: "#fca5a5" }}
        >
          {error}
        </p>
      )}

      {/* Summary of the best recommendation, only if we have data and no error. */}
      {!loading && !error && top && (
        <div style={{ marginTop: 16, fontSize: 14 }}>
          <p>
            <b>Best current recommendation:</b>{" "}
            <span style={{ fontWeight: 600 }}>{top.playType}</span>
          </p>
          <p className="muted">
            Predicted PPP <b>{top.PPP_pred.toFixed(3)}</b> (
            {(top.PPP_gap >= 0 ? "+" : "") + top.PPP_gap.toFixed(3)} PPP vs this
            opponent&apos;s average allowed).
          </p>
        </div>
      )}

      {/* Main table: full Top-K list of play types with PPP breakdowns. */}
      {!loading && !error && rows.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>#</th>
              <th>Play type</th>
              <th>Pred. PPP</th>
              <th>Our PPP (shrunk)</th>
              <th>Opp PPP allowed (shrunk)</th>
              <th>Gap vs opp avg</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              // Highlight the top-ranked row with a special CSS class "top1"
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

      {/* If there's no error but also no rows, show a friendly message. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No results yet. Try a different season or matchup.
        </p>
      )}

      {/* Show the exact backend URL being called so it's obvious how
          the frontend and backend are connected. Great to point at in defence. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/baseline?season={season}&amp;our={our}&amp;opp={opp}&amp;k={k}
        </code>
      </p>
    </section>
  );
}
