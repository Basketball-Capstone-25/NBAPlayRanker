// Loads offensive season profiles for two different teams.

"use client";

import React, { useEffect, useState } from "react";
import { teamComparisonKey, type TeamComparison, type TeamComparisonFilters } from "../../../domain/team-comparison";
import { fetchTeamComparison } from "../../../infrastructure/team-comparison";
import { withRequestDeadline } from "../../../infrastructure/playtype-dashboard";
import { TeamComparisonTable } from "./TeamComparisonTable";
import styles from "./matchup-dashboard.module.css";

type State = { data: TeamComparison | null; loading: boolean; error: string };
const initial = (): State => ({ data: null, loading: true, error: "" });

/** Render the two-team comparison only when the request succeeds and offer a retry if it fails. */
export function TeamComparisonResultState({ state, minimum, retry }: { state: State; minimum: number; retry(): void }) {
  if (state.loading) return <div className={styles.card} role="status">Loading team profiles…</div>;
  if (state.error) return <div className={styles.card} role="alert"><h2>Team profiles unavailable</h2>
    <p>{state.error}</p><button className={styles.button} type="button" onClick={retry}>Retry comparison</button></div>;
  return state.data ? <TeamComparisonTable data={state.data} minimum={minimum} /> : null;
}

function TeamComparisonLoader({ filters, minimum }: { filters: TeamComparisonFilters; minimum: number }) {
  const [state, setState] = useState(initial);
  const [attempt, setAttempt] = useState(0);
  const { season, team, opponent } = filters;
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState(initial());
    void withRequestDeadline(fetchTeamComparison({ season, team, opponent }, controller.signal),
      "The team comparison took too long. Please try again.")
      .then(data => { if (active) setState({ data, loading: false, error: "" }); })
      .catch(error => { if (active) setState({ data: null, loading: false,
        error: error instanceof Error ? error.message : "Could not load the comparison. Please try again." }); })
      .finally(() => controller.abort());
    return () => { active = false; controller.abort(); };
  }, [season, team, opponent, attempt]);
  return <TeamComparisonResultState state={state} minimum={minimum} retry={() => setAttempt(value => value + 1)} />;
}

/** Key the request by the ordered team pair and season so swapping teams also swaps the comparison direction. */
export default function TeamComparisonPanel({ filters, minimum }: { filters: TeamComparisonFilters; minimum: number }) {
  if (!filters.season || !filters.team || !filters.opponent)
    return <div className={styles.card} role="status">Select two teams and a season.</div>;
  if (filters.team === filters.opponent)
    return <div className={styles.card} role="status">Choose two different teams.</div>;
  // A changed selection gets fresh state before an earlier request can finish.
  return <TeamComparisonLoader key={teamComparisonKey(filters)} filters={filters} minimum={minimum} />;
}
