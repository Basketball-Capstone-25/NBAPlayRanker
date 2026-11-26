// Loads a team’s source outcome rates for one season.

"use client";

import React, { useEffect, useState } from "react";
import { outcomeKey, type OutcomeFilters, type PossessionOutcomes } from "../../../domain/possession-outcomes";
import { fetchPossessionOutcomes } from "../../../infrastructure/possession-outcomes";
import { withRequestDeadline } from "../../../infrastructure/playtype-dashboard";
import { PossessionOutcomesTable } from "./PossessionOutcomesTable";
import styles from "./matchup-dashboard.module.css";

type State = { data: PossessionOutcomes | null; loading: boolean; error: string };
const initial = (): State => ({ data: null, loading: true, error: "" });

/** Show loading and retry states before displaying the returned outcome rates. */
export function PossessionOutcomesResultState({ state, minimum, retry }: { state: State; minimum: number; retry(): void }) {
  if (state.loading) return <div className={styles.card} role="status">Loading possession rates…</div>;
  if (state.error) return <div className={styles.card} role="alert"><h2>Possession rates unavailable</h2>
    <p>{state.error}</p><button className={styles.button} type="button" onClick={retry}>Retry rates</button></div>;
  return state.data ? <PossessionOutcomesTable data={state.data} minimum={minimum} /> : null;
}

function PossessionOutcomesLoader({ filters, minimum }: { filters: OutcomeFilters; minimum: number }) {
  const [state, setState] = useState(initial);
  const [attempt, setAttempt] = useState(0);
  const { season, team } = filters;
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState(initial());
    void withRequestDeadline(fetchPossessionOutcomes({ season, team }, controller.signal),
      "The possession rates took too long. Please try again.")
      .then(data => { if (active) setState({ data, loading: false, error: "" }); })
      .catch(error => { if (active) setState({ data: null, loading: false,
        error: error instanceof Error ? error.message : "Could not load possession rates. Please try again." }); })
      .finally(() => controller.abort());
    return () => { active = false; controller.abort(); };
  }, [season, team, attempt]);
  return <PossessionOutcomesResultState state={state} minimum={minimum} retry={() => setAttempt(value => value + 1)} />;
}

/** Reset the request when the team or season changes so an earlier reply cannot replace the new selection. */
export default function PossessionOutcomesPanel({ filters, minimum }: { filters: OutcomeFilters; minimum: number }) {
  if (!filters.season || !filters.team)
    return <div className={styles.card} role="status">Select a team and a season.</div>;
  // Reset immediately when the selection changes, before an older request returns.
  return <PossessionOutcomesLoader key={outcomeKey(filters)} filters={filters} minimum={minimum} />;
}
