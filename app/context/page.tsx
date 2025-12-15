// app/context/page.tsx
//
// This file implements the Context Simulator screen
//
// Purpose:
// - Collect late-game context (score margin, period, time remaining) on the frontend.
// - Let the user choose season / our team / opponent.
// - Call a backend endpoint (`/rank-plays/context-ml`) that returns a **ranked Top-K list
//   of play types** with both a baseline PPP and a context-adjusted PPP.
// - Display the Top-3 recommendations
//


//client component because it uses React hooks
'use client';

import { useEffect, useMemo, useState } from "react";

//shared constants / helpers imported from the frontend utils module:
//API_BASE: base URL for our FastAPI server.
//SEASONS: list of seasons (strings) available in the data snapshot.
//TEAMS: list of team codes (strings) available in the data snapshot.
//contextRank: small helper function that builds the URL for

import { API_BASE, SEASONS, TEAMS, contextRank } from "../utils";

// Shape of a single ranked play returned by the context endpoint.
//
// The backend returns:
// - playType: label like "Spotup", "PRRollMan", etc.
// - basePPP: PPP from the baseline model (no context applied).
// - contextPPP: PPP after context adjustment (e.g., end-of-game effects).
// - deltaPPP: contextPPP - basePPP (how much the context changed the value).
// - rationale: human-readable sentence explaining the adjustment.
//
// We mirror that structure here for type-safety and clearer code.
type RankedPlay = {
  playType: string;
  basePPP: number;
  contextPPP: number;
  deltaPPP: number;
  rationale: string;
};

// Utility function to format a number of seconds into a scoreboard-style clock.
function formatClock(totalSeconds: number): string {
  // Guard: ensure non-negative integer seconds.
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Main React component for the Context Simulator page
export default function ContextSimulator() {

  // 1. State: matchup selection


  // Season dropdown:
  // Default to the **most recent** season in SEASONS to keep the demo current
  const [season, setSeason] = useState(SEASONS[SEASONS.length - 1]);

  // Team selection:
  // Default to TOR vs BOS to match the examples on other screens
  const [our, setOur] = useState("TOR");
  const [opp, setOpp] = useState("BOS");



  // 2. State: game context

  // Score margin: "our score − theirs".
  // Positive = we're winning, 0 = tied, negative = we're behind.
  const [scoreMargin, setScoreMargin] = useState(0);

  // Period (quarter / overtime).
  // We store it as a number for simplicity (1-4 = Q1–Q4, 5 = OT).
  const [period, setPeriod] = useState(4);

  // Time remaining in the current period (seconds).
  // Default is 120 (i.e., 2:00 left) to show a late-game scenario by default.
  const [timeRemaining, setTimeRemaining] = useState(120);


  // 3. State: data from backend

  // Rows returned by the context endpoint (Top-3 ranked plays).
  const [rows, setRows] = useState<RankedPlay[]>([]);

  // Loading and error flags so the UI can show progress / issues.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 4. Effect: call backend whenever matchup or context change

  // This effect is the engine of the page
  // Whenever season, teams, score margin, period, or time remaining change,
  // we:
  //   1. set loading=true,
  //   2. call the contextRank helper,
  //   3. update rows / error based on the response,
  //   4. set loading=false.
  //

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // Build the request payload for the context endpoint.

        const result = await contextRank({
          season,
          our,
          opp,
          margin: scoreMargin,
          period,
          timeRemaining,
          k: 3, // We only show the Top-3 plays on this screen to keep it focused.
        });

        if (!cancelled) {
          // We trust the backend to return an array of RankedPlay-like objects.
          setRows(result as RankedPlay[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          // Surface a friendly error message in the UI.
          setError(err?.message ?? "Unable to load context ML results.");
          setRows([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    run();

    // Cleanup function: marks the request as cancelled if the component unmounts.
    return () => {
      cancelled = true;
    };
  }, [
    season,
    our,
    opp,
    scoreMargin,
    period,
    timeRemaining,
    // Dependency array: whenever any of these change, we re-run the effect.
  ]);


  // 5. Derived text: human-readable description of the scenario


  // We use useMemo so this string is recomputed only when the
  // underlying inputs change, not on every render.

  const scenarioText = useMemo(() => {
    const clock = formatClock(timeRemaining);

    // Build the "margin" phrase.
    let marginPart = "tied";
    if (scoreMargin > 0) {
      marginPart = `up ${scoreMargin}`;
    } else if (scoreMargin < 0) {
      marginPart = `down ${Math.abs(scoreMargin)}`;
    }

    // Example outputs:
    // "Scenario: tied in Q4 with 2:00 left."
    // "Scenario: up 3 in Q4 with 0:27 left."
    return `Scenario: ${marginPart} in Q${period} with ${clock} left.`;
  }, [scoreMargin, period, timeRemaining]);


  // 6. Render: JSX for the Context Simulator page


  return (
    <section className="card">
      {/* Page title with a badge that reminds the reader this is
          a working UI backed by baseline-style logic in this phase. */}
      <h1 className="h1">
        Context Simulator <span className="badge">UI mock + baseline</span>
      </h1>

      {/* High-level explanation for the defence and for users. */}
      <p className="muted">
        This screen shows where the future context-aware ML model will live. Today it
        still calls a FastAPI context endpoint that wraps the baseline ranking, but it
        already passes full game context so we can talk through late-game decision
        making and plug in a richer model in a later phase without changing the UI.
      </p>

      {/* -------------------- */}
      {/* Matchup / context form */}
      {/* -------------------- */}
      <div className="form-grid" style={{ marginTop: 16 }}>
        {/* Season selector: chooses which snapshot of Synergy data to use. */}
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

        {/* Our team selector:
            - When "our" changes, we also ensure opponent stays different to
              avoid a degenerate our=opp matchup. */}
        <label>
          Our team
          <select
            className="input"
            value={our}
            onChange={(e) => {
              const next = e.target.value;
              setOur(next);

              // If the user picks the same team for both sides, automatically
              // adjust the opponent to the first *different* team as a fallback.
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
            - We filter out "our" so the dropdown never shows the same team twice. */}
        <label>
          Opponent
          <select
            className="input"
            value={opp}
            onChange={(e) => setOpp(e.target.value)}
          >
            {TEAMS.filter((t) => t !== our).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        {/* Score margin input:
            - Uses "our score − theirs".
            - Stored as a number in state, clamped via Number(...) || 0. */}
        <label>
          Score margin (our score − theirs)
          <input
            className="input"
            type="number"
            value={scoreMargin}
            onChange={(e) => setScoreMargin(Number(e.target.value) || 0)}
          />
        </label>

        {/* Period selector:
            - Q1–Q4 plus OT, encoded as numbers 1–5.
            - Keeps the UI consistent with basketball terminology. */}
        <label>
          Period
          <select
            className="input"
            value={period}
            onChange={(e) => setPeriod(Number(e.target.value) || 1)}
          >
            <option value={1}>Q1</option>
            <option value={2}>Q2</option>
            <option value={3}>Q3</option>
            <option value={4}>Q4</option>
            <option value={5}>OT</option>
          </select>
        </label>

        {/* Time remaining input:
            - Stored in seconds, clamped between 0 and 720 (full 12-minute period).
            - Used to build the scenario text and sent to the backend. */}
        <label>
          Time remaining (seconds)
          <input
            className="input"
            type="number"
            min={0}
            max={720}
            value={timeRemaining}
            onChange={(e) =>
              setTimeRemaining(
                Math.max(0, Math.min(720, Number(e.target.value) || 0))
              )
            }
          />
        </label>
      </div>

      {/* Human-readable scenario text that ties all the context together. */}
      <p className="muted" style={{ marginTop: 16 }}>
        {scenarioText} Below is the same Top 3 baseline ranking you saw on the other
        screens, reused here as a placeholder until the ML model is integrated.
      </p>

      {/* ------------- */}
      {/* Loading / error */}
      {/* ------------- */}

      {loading && (
        <p className="muted" style={{ marginTop: 16 }}>
          Running baseline model…
        </p>
      )}

      {error && !loading && (
        <p
          className="muted"
          style={{ marginTop: 16, color: "#fca5a5" }}
        >
          {error}
        </p>
      )}

      {/* ------------- */}
      {/* Results list   */}
      {/* ------------- */}

      {/* When we have non-empty rows and no errors, show the Top-3 ranked plays
          as an ordered list.

          Each item shows:
          - play type name,
          - base PPP → context PPP (+/- delta),
          - a one-line rationale explaining *why* context changed that play. */}
      {!loading && !error && rows.length > 0 && (
        <ol style={{ marginTop: 16, paddingLeft: 20 }}>
          {rows.map((d) => (
            <li key={d.playType} style={{ marginBottom: 8 }}>
              <div>
                <b>{d.playType}</b>{" "}
                <span className="muted" style={{ fontSize: 12 }}>
                  base {d.basePPP.toFixed(3)} PPP → context{" "}
                  {d.contextPPP.toFixed(3)} PPP (
                  {d.deltaPPP >= 0 ? "+" : ""}
                  {d.deltaPPP.toFixed(3)})
                </span>
              </div>
              <div className="muted" style={{ fontSize: 11 }}>
                {d.rationale}
              </div>
            </li>
          ))}
        </ol>
      )}

      {/* Fallback when the request succeeded but didn't return any rows. */}
      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No baseline results yet. Try changing the season or teams.
        </p>
      )}

      {/* ------------- */}
      {/* API call trace */}
      {/* ------------- */}

      {/* For defence and debugging, we print the exact query that the contextRank
          helper builds. This shows that the frontend is passing all the context
          fields to the backend, and it documents the contract for the
          `/rank-plays/context-ml` endpoint. */}
      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/context-ml
          ?season={season}
          &amp;our={our}
          &amp;opp={opp}
          &amp;margin={scoreMargin}
          &amp;period={period}
          &amp;time_remaining={timeRemaining}
          &amp;k=3
        </code>
      </p>
    </section>
  );
}
