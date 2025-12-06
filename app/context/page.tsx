'use client';

import { useEffect, useMemo, useState } from "react";
import { API_BASE, SEASONS, TEAMS, baselineRank } from "../utils";

type RankedPlay = {
  playType: string;
  PPP_pred: number;
  PPP_off: number;
  PPP_def: number;
  PPP_gap: number;
};

function formatClock(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ContextSimulator() {
  const [season, setSeason] = useState(SEASONS[SEASONS.length - 1]);
  const [our, setOur] = useState("TOR");
  const [opp, setOpp] = useState("BOS");

  const [scoreMargin, setScoreMargin] = useState(0); // our score − theirs
  const [period, setPeriod] = useState(4);
  const [timeRemaining, setTimeRemaining] = useState(120); // seconds

  const [rows, setRows] = useState<RankedPlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        setLoading(true);
        setError(null);
        // For now this still calls the same baseline endpoint with K=3.
        const result = await baselineRank({ season, our, opp, k: 3 });
        if (!cancelled) {
          setRows(result as RankedPlay[]);
        }
      } catch (err: any) {
        console.error(err);
        if (!cancelled) {
          setError(err?.message ?? "Unable to load baseline results.");
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
  }, [season, our, opp]);

  const scenarioText = useMemo(() => {
    const clock = formatClock(timeRemaining);
    let marginPart = "tied";
    if (scoreMargin > 0) {
      marginPart = `up ${scoreMargin}`;
    } else if (scoreMargin < 0) {
      marginPart = `down ${Math.abs(scoreMargin)}`;
    }
    return `Scenario: ${marginPart} in Q${period} with ${clock} left.`;
  }, [scoreMargin, period, timeRemaining]);

  return (
    <section className="card">
      <h1 className="h1">
        Context Simulator <span className="badge">UI mock + baseline</span>
      </h1>
      <p className="muted">
        This screen shows where the future context-aware ML model will live. Today it
        still calls the same FastAPI baseline endpoint but wraps it in extra game
        context sliders so you can talk through late-game decision making.
      </p>

      <div className="form-grid" style={{ marginTop: 16 }}>
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

        <label>
          Our team
          <select
            className="input"
            value={our}
            onChange={(e) => {
              const next = e.target.value;
              setOur(next);
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

        <label>
          Score margin (our score − theirs)
          <input
            className="input"
            type="number"
            value={scoreMargin}
            onChange={(e) => setScoreMargin(Number(e.target.value) || 0)}
          />
        </label>

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

        <label>
          Time remaining (seconds)
          <input
            className="input"
            type="number"
            min={0}
            max={720}
            value={timeRemaining}
            onChange={(e) =>
              setTimeRemaining(Math.max(0, Math.min(720, Number(e.target.value) || 0)))
            }
          />
        </label>
      </div>

      <p className="muted" style={{ marginTop: 16 }}>
        {scenarioText} Below is the same Top 3 baseline ranking you saw on the other
        screens, reused here as a placeholder until the ML model is integrated.
      </p>

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

      {!loading && !error && rows.length > 0 && (
        <ol style={{ marginTop: 16, paddingLeft: 20 }}>
          {rows.map((d) => (
            <li key={d.playType} style={{ marginBottom: 4 }}>
              <b>{d.playType}</b>:{" "}
              {(d.PPP_gap >= 0 ? "+" : "") + d.PPP_gap.toFixed(3)} PPP vs this
              opponent&apos;s average allowed.
            </li>
          ))}
        </ol>
      )}

      {!loading && !error && rows.length === 0 && (
        <p className="muted" style={{ marginTop: 16 }}>
          No baseline results yet. Try changing the season or teams.
        </p>
      )}

      <p className="muted" style={{ marginTop: 16, fontSize: 11 }}>
        API call:&nbsp;
        <code>
          {API_BASE}/rank-plays/baseline?season={season}&amp;our={our}&amp;opp={opp}&amp;k=3
        </code>
      </p>
    </section>
  );
}
