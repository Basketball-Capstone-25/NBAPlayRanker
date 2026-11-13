"use client";

import React, { useEffect, useState } from "react";
import { trendKey, type SeasonTrend, type TrendFilters } from "../../../domain/season-trend";
import { playtypeName } from "../../../domain/playtype-dashboard";
import { fetchSeasonTrend } from "../../../infrastructure/season-trend";
import { withRequestDeadline } from "../../../infrastructure/playtype-dashboard";
import { SeasonTrendChart, type TrendMetric } from "./SeasonTrendChart";
import { SeasonTrendTable } from "./SeasonTrendTable";
import styles from "./matchup-dashboard.module.css";

type State = { data: SeasonTrend | null; loading: boolean; error: string };
const initial = (): State => ({ data: null, loading: true, error: "" });

function TrendView({ data, minimum }: { data: SeasonTrend; minimum: number }) {
  const [metric, setMetric] = useState<TrendMetric>("ppp");
  return <section className={styles.card} aria-labelledby="trend-title">
    <h2 id="trend-title">{data.filters.team}: {playtypeName(data.filters.play_type)} across seasons</h2>
    <p className={styles.note}>Historical offensive regular-season totals. Missing seasons stay as gaps.</p>
    {data.status === "no_data" ? <p role="status">No recorded possessions for this selection across the available seasons.</p> : null}
    <div className={styles.tabs} role="group" aria-label="Trend metric">
      <button type="button" aria-pressed={metric === "ppp"} onClick={() => setMetric("ppp")}>PPP</button>
      <button type="button" aria-pressed={metric === "usage_pct"} onClick={() => setMetric("usage_pct")}>Usage</button>
    </div>
    <SeasonTrendChart data={data} metric={metric} minimum={minimum} />
    <p className={styles.note}>Hollow circles mark fewer than {minimum.toLocaleString("en-US")} possessions. The cutoff changes warnings, not the values.</p>
    <SeasonTrendTable data={data} minimum={minimum} />
    <details className={styles.details}><summary>How this trend is calculated</summary>
      <p>PPP = play-type points ÷ play-type possessions. Usage = play-type possessions ÷ all recorded team offensive possessions that season × 100.</p>
      <p>Each season uses the same definitions. A missing category stays unavailable; a recorded zero has 0 usage when team possessions are positive and PPP is unavailable.</p>
      <p>These season totals do not describe recent-game form or predict future results.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}

export function TrendResultState({ state, minimum, retry }: { state: State; minimum: number; retry(): void }) {
  if (state.loading) return <div className={styles.card} role="status">Loading season trend…</div>;
  if (state.error) return <div className={styles.card} role="alert"><h2>Season trend unavailable</h2>
    <p>{state.error}</p><button className={styles.button} type="button" onClick={retry}>Retry trend</button></div>;
  return state.data ? <TrendView data={state.data} minimum={minimum} /> : null;
}

function TrendLoader({ filters, minimum }: { filters: TrendFilters; minimum: number }) {
  const [state, setState] = useState(initial);
  const [attempt, setAttempt] = useState(0);
  const { team, play_type } = filters;
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState(initial());
    void withRequestDeadline(fetchSeasonTrend({ team, play_type }, controller.signal),
      "The season trend took too long. Please try again.")
      .then(data => { if (active) setState({ data, loading: false, error: "" }); })
      .catch(error => { if (active) setState({ data: null, loading: false,
        error: error instanceof Error ? error.message : "Could not load the trend. Please try again." }); })
      .finally(() => controller.abort());
    return () => { active = false; controller.abort(); };
  }, [team, play_type, attempt]);
  return <TrendResultState state={state} minimum={minimum} retry={() => setAttempt(value => value + 1)} />;
}

export default function SeasonTrendPanel({ filters, minimum = 20 }: { filters: TrendFilters; minimum?: number }) {
  if (!filters.team || !filters.play_type) return <div className={styles.card} role="status">Select a team and play type.</div>;
  return <TrendLoader key={trendKey(filters)} filters={filters} minimum={minimum} />;
}
