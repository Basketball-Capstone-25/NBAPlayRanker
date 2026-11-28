// Loads the history of one offensive play type for one team.

"use client";

import React, { useEffect, useState } from "react";
import { trendKey, type SeasonTrend, type TrendFilters } from "../../../domain/season-trend";
import { playtypeName } from "../../../domain/playtype-dashboard";
import { trendChange } from "../../../domain/trend-display";
import { fetchSeasonTrend } from "../../../infrastructure/season-trend";
import { withRequestDeadline } from "../../../infrastructure/playtype-dashboard";
import { SeasonTrendChart, type TrendMetric } from "./SeasonTrendChart";
import { SeasonTrendTable } from "./SeasonTrendTable";
import styles from "./matchup-dashboard.module.css";
import trendStyles from "./trend-display.module.css";

type State = { data: SeasonTrend | null; loading: boolean; error: string };
const initial = (): State => ({ data: null, loading: true, error: "" });

function TrendView({ data, minimum }: { data: SeasonTrend; minimum: number }) {
  const [metric, setMetric] = useState<TrendMetric>("ppp");
  const [startAtZero, setStartAtZero] = useState(false);
  const change = trendChange(data.rows, metric, minimum);
  const play = playtypeName(data.filters.play_type);
  const partialSeason = data.rows.some(row => row.season === "2026-27");
  const changeText = !change ? "One season cannot show a change."
    : change.smallSample ? "Small sample: hold off on a takeaway."
    : change.direction === "same" ? metric === "ppp" ? "Scoring was about the same." : "Usage was about the same."
    : metric === "ppp" ? `${change.displayedDifference.toFixed(1)} ${change.direction === "up" ? "more" : "fewer"} points per 100 ${play} possessions.`
    : `${change.displayedDifference.toFixed(1)} percentage points ${change.direction === "up" ? "more" : "less"} of ${data.filters.team}’s recorded offense.`;
  return <section className={styles.card} aria-labelledby="trend-title">
    <h2 id="trend-title">Has {data.filters.team}’s {play} changed?</h2>
    <p className={styles.note}>Compare scoring and usage across all available regular seasons. This view follows one team; it does not use the selected opponent.</p>
    {data.status === "no_data" ? <p role="status">No recorded possessions for this selection across the available seasons.</p> : null}
    {data.status !== "no_data" ? <div className={trendStyles.summary} aria-live="polite">
      {change ? <p className={trendStyles.period}>{change.first.season} → {change.last.season} · first and last available seasons</p> : null}
      <p className={trendStyles.headline}>{changeText}</p>
      {change ? <div className={trendStyles.endpoints}>{[change.first, change.last].map(row => <div key={row.season} className={trendStyles.endpoint}>
        <span>{row.season}</span><strong>{row[metric]!.toFixed(metric === "ppp" ? 3 : 1)}{metric === "ppp" ? " PPP" : "% usage"}</strong>
        <span>{row.possessions?.toLocaleString("en-US")} play possessions</span>
      </div>)}</div> : null}
      <p className={trendStyles.explanation}>{change?.smallSample
        ? `The first or last season has fewer than ${minimum.toLocaleString("en-US")} possessions. Treat the difference cautiously.`
        : metric === "ppp" ? "This shows how scoring on this play changed. It does not predict extra team points per game."
          : "This shows how much of the team’s recorded offense used this play, not whether using it more would work."}</p>
      {change?.missingBetween ? <p className={trendStyles.explanation}>Some seasons in between are missing. The line leaves those gaps open.</p> : null}
      {partialSeason ? <p className={trendStyles.partial}>2026–27 is a partial season in this dataset. Its rates can still change as more games are played.</p> : null}
    </div> : null}
    <div className={trendStyles.controls}>
      <div className={styles.tabs} role="group" aria-label="Trend metric">
        <button type="button" aria-pressed={metric === "ppp"} onClick={() => setMetric("ppp")}>Scoring (PPP)</button>
        <button type="button" aria-pressed={metric === "usage_pct"} onClick={() => setMetric("usage_pct")}>Usage</button>
      </div>
      <label className={trendStyles.scaleControl}><input type="checkbox" checked={startAtZero} onChange={event => setStartAtZero(event.target.checked)} />Start chart at zero</label>
    </div>
    <SeasonTrendChart data={data} metric={metric} minimum={minimum} scale={startAtZero ? "zero" : "focused"} />
    <p className={styles.note}>Hollow circles mark fewer than {minimum.toLocaleString("en-US")} possessions. Missing seasons stay as gaps.</p>
    <details className={styles.details}><summary>View exact season values</summary><SeasonTrendTable data={data} minimum={minimum} /></details>
    <details className={styles.details}><summary>How this trend is calculated</summary>
      <p>PPP = play-type points ÷ play-type possessions. Usage = play-type possessions ÷ all recorded team offensive possessions that season × 100.</p>
      <p>Usage covers the qualifying rows in the source. It is not a verified share of every game possession.</p>
      <p>Scoring change = (last recorded PPP − first recorded PPP) × 100, in points per 100 possessions of this play. Usage change = last usage % − first usage %, in percentage points.</p>
      <p>Each season uses the same definitions. A missing category stays unavailable; a recorded zero has 0 usage when team possessions are positive and PPP is unavailable.</p>
      <p>These season totals do not describe recent-game form or predict future results.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}

/** Offer a retry on failure. Keep a successful response with missing seasons available for inspection. */
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

/** Reload when the team or play type changes. This view spans seasons and needs no opponent. */
export default function SeasonTrendPanel({ filters, minimum = 20 }: { filters: TrendFilters; minimum?: number }) {
  if (!filters.team || !filters.play_type) return <div className={styles.card} role="status">Select a team and play type.</div>;
  return <TrendLoader key={trendKey(filters)} filters={filters} minimum={minimum} />;
}
