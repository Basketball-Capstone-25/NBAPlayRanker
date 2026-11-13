"use client";

import React, { useEffect, useState } from "react";
import { leagueKey, type LeagueComparison, type LeagueFilters } from "../../../domain/league-comparison";
import { fetchLeagueComparison } from "../../../infrastructure/league-comparison";
import { withRequestDeadline } from "../../../infrastructure/playtype-dashboard";
import { LeagueComparisonTable } from "./LeagueComparisonTable";
import styles from "./matchup-dashboard.module.css";

type State = { data: LeagueComparison | null; loading: boolean; error: string };
const initial = (): State => ({ data: null, loading: true, error: "" });

export function LeagueResultState({ state, minimum, retry }: { state: State; minimum: number; retry(): void }) {
  if (state.loading) return <div className={styles.card} role="status">Loading league comparison…</div>;
  if (state.error) return <div className={styles.card} role="alert"><h2>League comparison unavailable</h2>
    <p>{state.error}</p><button className={styles.button} type="button" onClick={retry}>Retry comparison</button></div>;
  return state.data ? <LeagueComparisonTable data={state.data} minimum={minimum} /> : null;
}

function LeagueLoader({ filters, minimum }: { filters: LeagueFilters; minimum: number }) {
  const [state, setState] = useState(initial);
  const [attempt, setAttempt] = useState(0);
  const { season, team, side } = filters;
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState(initial());
    void withRequestDeadline(fetchLeagueComparison({ season, team, side }, controller.signal),
      "The league comparison took too long. Please try again.")
      .then(data => { if (active) setState({ data, loading: false, error: "" }); })
      .catch(error => { if (active) setState({ data: null, loading: false,
        error: error instanceof Error ? error.message : "Could not load the comparison. Please try again." }); })
      .finally(() => controller.abort());
    return () => { active = false; controller.abort(); };
  }, [season, team, side, attempt]);
  return <LeagueResultState state={state} minimum={minimum} retry={() => setAttempt(value => value + 1)} />;
}

export default function LeagueComparisonPanel({ filters, minimum }: { filters: LeagueFilters; minimum: number }) {
  if (!filters.season || !filters.team) return <div className={styles.card} role="status">Select a team and season.</div>;
  // A changed selection gets fresh state before rendering, even if an old reply arrives late.
  return <LeagueLoader key={leagueKey(filters)} filters={filters} minimum={minimum} />;
}
