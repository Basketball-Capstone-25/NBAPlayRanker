"use client";

// UI Layer — Timeout Quick-Call Mode (SCRUM-628).
// Top 3 plays in large text, each with its best shot zone and a one-line reason.
// Score, period and clock change with one tap; the list refreshes within 2 seconds.

import Link from "next/link";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  LATE_GAME_SCENARIOS,
  applyScenario,
  canStep,
  formatClock,
  formatMargin,
  formatPeriod,
  situationFromParams,
  stepSituation,
  type QuickCallSituation,
  type SituationField,
} from "../../../domain/quick-call";
import { fetchQuickCall, QuickCallForbiddenError, type QuickCallResponse } from "../../../services/quick-call";
import styles from "./QuickCallMode.module.css";

const REFRESH_DEBOUNCE_MS = 250;

type Status = "loading" | "ready" | "error" | "forbidden";

const CONTROLS: { field: SituationField; label: string; show: (s: QuickCallSituation) => string }[] = [
  { field: "margin", label: "Score", show: (s) => formatMargin(s.margin) },
  { field: "period", label: "Period", show: (s) => formatPeriod(s.period) },
  { field: "timeRemaining", label: "Clock", show: (s) => formatClock(s.timeRemaining) },
];

function QuickCallInner() {
  const params = useSearchParams();
  const [situation, setSituation] = useState<QuickCallSituation>(() => situationFromParams(params));
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [result, setResult] = useState<QuickCallResponse | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState("");
  const [lastRefreshMs, setLastRefreshMs] = useState<number | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const hasMatchup = Boolean(situation.season && situation.ourTeam && situation.oppTeam);

  useEffect(() => {
    if (!hasMatchup) {
      setStatus("error");
      setMessage("Open Quick-Call Mode from the Context ML page so it knows the matchup.");
      return;
    }
    const timer = setTimeout(() => void refresh(situation), REFRESH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [situation, hasMatchup]);

  useEffect(() => () => inFlight.current?.abort(), []);

  async function refresh(s: QuickCallSituation) {
    inFlight.current?.abort(); // a newer tap always wins
    const controller = new AbortController();
    inFlight.current = controller;
    setStatus((prev) => (prev === "ready" ? "ready" : "loading"));
    const started = performance.now();
    try {
      const response = await fetchQuickCall(s, controller.signal);
      if (controller.signal.aborted) return;
      setResult(response);
      setStatus("ready");
      setMessage("");
      setLastRefreshMs(Math.round(performance.now() - started));
    } catch (error) {
      if (controller.signal.aborted) return;
      setStatus(error instanceof QuickCallForbiddenError ? "forbidden" : "error");
      setMessage(error instanceof Error ? error.message : "Could not load quick calls.");
    }
  }

  function step(field: SituationField, direction: 1 | -1) {
    setScenarioId(null);
    setSituation((prev) => stepSituation(prev, field, direction));
  }

  function pickScenario(id: string) {
    const scenario = LATE_GAME_SCENARIOS.find((item) => item.id === id);
    if (!scenario) return;
    setScenarioId(id);
    setSituation((prev) => applyScenario(prev, scenario));
  }

  const contextHref = useMemo(() => {
    const query = new URLSearchParams({ season: situation.season, our: situation.ourTeam, opp: situation.oppTeam });
    return `/context?${query}`;
  }, [situation.season, situation.ourTeam, situation.oppTeam]);

  return (
    <main className={styles.page} aria-label="Timeout Quick-Call Mode">
      <div className={styles.top}>
        <div>
          <h1 className={styles.title}>Quick-Call</h1>
          <p className={styles.matchup}>
            {hasMatchup ? `${situation.ourTeam} vs ${situation.oppTeam} • ${situation.season}` : "No matchup selected"}
          </p>
        </div>
        <Link className={styles.backLink} href={contextHref}>
          Back to Context ML
        </Link>
      </div>

      <section className={styles.controls} aria-label="Game situation">
        {CONTROLS.map(({ field, label, show }) => (
          <div className={styles.control} key={field}>
            <span className={styles.controlLabel}>{label}</span>
            <button
              type="button"
              className={styles.stepButton}
              aria-label={`${label} down`}
              disabled={!canStep(situation, field, -1)}
              onClick={() => step(field, -1)}
            >
              −
            </button>
            <span className={styles.controlValue} data-testid={`value-${field}`}>
              {show(situation)}
            </span>
            <button
              type="button"
              className={styles.stepButton}
              aria-label={`${label} up`}
              disabled={!canStep(situation, field, 1)}
              onClick={() => step(field, 1)}
            >
              +
            </button>
          </div>
        ))}
      </section>

      <section className={styles.scenarios} aria-label="Late-game scenarios">
        {LATE_GAME_SCENARIOS.map((scenario) => (
          <button
            type="button"
            key={scenario.id}
            className={`${styles.scenario} ${scenarioId === scenario.id ? styles.scenarioActive : ""}`}
            aria-pressed={scenarioId === scenario.id}
            onClick={() => pickScenario(scenario.id)}
          >
            {scenario.label}
          </button>
        ))}
      </section>

      <div className={styles.status} role="status" aria-live="polite">
        {status === "loading" ? "Loading the top 3 plays…" : null}
        {status === "ready" && lastRefreshMs !== null
          ? `${result?.source === "baseline" ? "Baseline plays" : "Context-aware plays"} • updated in ${lastRefreshMs} ms`
          : null}
      </div>

      {status === "ready" && result?.source === "baseline" ? <div className={styles.notice}>{result.notice}</div> : null}
      {status === "error" || status === "forbidden" ? <div className={styles.error}>{message}</div> : null}

      {result && status !== "forbidden" ? (
        <ol className={styles.plays} aria-label="Top 3 plays">
          {result.plays.map((play) => (
            <li className={styles.play} key={play.play_type}>
              <span className={styles.rank}>{play.rank}</span>
              <div>
                <div className={styles.playName}>{play.play_type}</div>
                <div className={styles.zone}>Shot zone: {play.shot_zone}</div>
                <div className={styles.reason}>{play.reason}</div>
              </div>
              <div className={styles.ppp}>
                {play.ppp.toFixed(2)}
                <span className={styles.pppLabel}>expected PPP</span>
              </div>
            </li>
          ))}
        </ol>
      ) : null}
    </main>
  );
}

export default function QuickCallModeClient() {
  return (
    <Suspense fallback={<div style={{ padding: 16 }}>Loading…</div>}>
      <QuickCallInner />
    </Suspense>
  );
}
