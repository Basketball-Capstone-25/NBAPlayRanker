'use client';

import { useEffect, useState } from "react";
import { API_BASE, SEASONS, TEAMS, baselineRank } from "../utils";

// Minimal shape we care about for the raw preview.
// The backend may return extra fields (e.g., PPP_pred, PPP_gap) but we ignore them here.
type RankedPlay = {
  playType: string;
  PPP_off: number;
  PPP_def: number;
};

export default function DataExplorerPage() {
  // Filters: season + offense team + opponent
  const [season, setSeason] = useState(SEASONS[0]);
  const [team, setTeam] = useState("TOR");
  const [opp, setOpp] = useState("BOS");

  // Data + UI state
  const [rows, setRows] = useState<RankedPlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Whenever filters change, fetch a fresh preview of the underlying data.
  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // IMPORTANT: k MUST be <= 10 to match FastAPI validation.
        // We use Top 10 rows here as a small preview table.
        const result = await baselineRank({
          season,
          our: team,
          opp,
          k: 10,
        });

        if (!cancelled) {
          setRows(Array.isArray(result) ? (result as RankedPlay[]) : []);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
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

    return () => {
      cancelled = true;
    };
  }, [season, team, opp]);

  // CSV export URL for the current filters
  // NOTE: k=10 here as well (must be <= 10).
  const csvUrl = `${API_BASE}/rank-plays/baseline.csv?season=${encodeURIComponent(
    season
  )}&our=${encodeURIComponent(team)}&opp=${encodeURIComponent(opp)}&k=10`;

  return (
    <section className="card">
      <h1 className="h1">Data Explorer</h1>
      <p className="muted">
        Preview the underlying team–season–play-type matchup data that feeds the baseline
        model. Analysts can export the current view as CSV and do deeper work in Excel, R,
        or Python.
      </p>

      {/* Filter form: season, offensive team, and opponent. */}
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

        {/* Offensive team selector */}
        <label>
          Offense (our team)
          <select
            className="input"
            value={team}
            onChange={(e) => {
              const next = e.target.value;
              setTeam(next);
              // Make sure offense and defense are not the same team.
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

        {/* Opponent selector */}
        <label>
          Opponent
          <select
            className="input"
            value={opp}
            onChange={(e) => {
              const next = e.target.value;
              setOpp(next);
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

      {/* Small bar for context + CSV export */}
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
        <p className="muted" style={{ fontSize: 12 }}>
          Showing baseline input stats for <b>{team}</b> vs <b>{opp}</b> in{" "}
          <b>{season}</b>.
        </p>
        <a
          href={csvUrl}
          className="btn"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download CSV
        </a>
      </div>

      {/* Loading / error / empty states */}
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

      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No rows returned for this combination yet.
        </p>
      )}

      {/* Main data table – NO predicted / gap columns, just raw PPP inputs */}
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
                  <td>{idx + 1}</td>
                  <td>{r.playType}</td>
                  <td>{r.PPP_off?.toFixed(3)}</td>
                  <td>{r.PPP_def?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Show the exact API call being made (for defence explanation). */}
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
