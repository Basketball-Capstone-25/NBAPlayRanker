"use client";

import React, { useEffect, useRef, useState } from "react";
import { playtypeName } from "../../../domain/playtype-dashboard";
import { summaryKey, type SummaryFilters, type SeasonSummary as SummaryData } from "../../../domain/season-summary";
import { emptySummary, fetchSeasonSummary, selectSummary, SummaryRequest, type SummaryState } from "../../../infrastructure/season-summary";
import styles from "./season-summary.module.css";

const count = (value: number) => value.toLocaleString("en-US", { maximumFractionDigits: 0 });
const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

export function SeasonSummaryResults({ data }: { data: SummaryData }) {
  if (data.status === "no_data") return <p role="status">No data for this period. Try another season or matchup.</p>;
  const { summary, filters } = data;
  return <>
    <p className={styles.context}>{filters.our} vs {filters.opp} · {filters.season} regular season · Top {summary.k_returned} play types</p>
    <dl className={styles.metrics}>
      <div><dt>Observed team PPP</dt><dd>{summary.observed_team_ppp.toFixed(3)}</dd></div>
      <div><dt>Modeled Top-K PPP</dt><dd>{summary.modeled_topk_ppp.toFixed(3)}</dd></div>
      <div><dt>Modeled difference</dt><dd>{signed(summary.modeled_uplift_ppp)} <small>PPP</small></dd></div>
    </dl>
    <p className={styles.note}>Historical comparison, not measured improvement or a forecast.</p>
    <h3>How often these plays were used</h3>
    <p className={styles.context}>Together: {summary.selected_usage_pct.toFixed(1)}% of {filters.our}’s {count(summary.team_possessions)} recorded offensive play-type possessions.</p>
    <ul className={styles.usage} aria-label="Selected play types and season usage">
      {data.play_types.map(row => <li key={row.play_type}>
        <span>{row.rank}. {playtypeName(row.play_type)}</span>
        <span className={styles.track} aria-hidden="true"><span style={{ width: `${row.usage_pct}%` }} /></span>
        <strong>{row.usage_pct.toFixed(1)}%</strong>
        <span className={styles.possessions}>{count(row.possessions)} poss.</span>
      </li>)}
    </ul>
    <details className={styles.details}><summary>What do these numbers mean?</summary>
      <p>PPP means points per possession. Observed PPP uses the team’s recorded season totals. Modeled Top-K PPP averages the selected baseline scores using their historical possession shares.</p>
      <p>Difference = modeled Top-K PPP − observed team PPP. Usage = play-type possessions ÷ all recorded offensive play-type possessions × 100.</p>
      <p>Baseline weights: {(filters.w_off * 100).toFixed(1)}% offense and {(filters.w_def * 100).toFixed(1)}% opponent defense. Small samples are adjusted toward league averages.</p>
      <p>Season totals only; last-5 and last-10-game windows are unavailable. These are team-season profiles, not games played directly against each other.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </>;
}

export function SeasonSummaryState({ state, retry }: { state: SummaryState; retry: () => void }) {
  if (state.loading) return <p role="status">Loading season summary…</p>;
  if (state.error) return <div role="alert"><p>{state.error}</p><button type="button" className={styles.button} onClick={retry}>Retry summary</button></div>;
  return state.data ? <SeasonSummaryResults data={state.data} /> : null;
}

function SummaryLoader({ filters }: { filters: SummaryFilters }) {
  const [state, setState] = useState(emptySummary);
  const [attempt, setAttempt] = useState(0);
  const request = useRef<SummaryRequest | null>(null);
  if (!request.current) request.current = new SummaryRequest();
  const { season, our, opp, k, w_off } = filters;
  useEffect(() => {
    const current = request.current!;
    void current.run({ season, our, opp, k, w_off }, fetchSeasonSummary, setState);
    return () => current.cancel();
  }, [season, our, opp, k, w_off, attempt]);
  return <SeasonSummaryState state={selectSummary(state, filters)} retry={() => setAttempt(value => value + 1)} />;
}

export default function SeasonSummary({ filters, unavailableReason = "Choose a season and two different teams." }: {
  filters: SummaryFilters | null; unavailableReason?: string;
}) {
  const valid = filters && filters.season && filters.our && filters.opp && filters.our !== filters.opp
    && Number.isInteger(filters.k) && filters.k >= 1 && filters.k <= 10
    && Number.isFinite(filters.w_off) && filters.w_off >= 0 && filters.w_off <= 1;
  return <section className={styles.card} aria-label="Season summary">
    <h2>Season summary</h2>
    {valid ? <SummaryLoader key={summaryKey(filters)} filters={filters} /> : <p role="status">{unavailableReason}</p>}
  </section>;
}
