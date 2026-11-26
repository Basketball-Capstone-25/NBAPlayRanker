// Team and league rates for the same season, side and play type.

"use client";

import React, { useId, useState } from "react";
import type { LeagueComparison, LeagueRate, LeagueSample } from "../../../domain/league-comparison";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";
import readout from "./comparison-readouts.module.css";

const metrics: { key: LeagueRate; label: string; detail: string }[] = [
  { key: "fg_pct", label: "Shooting (FG%)", detail: "Shots made out of every 100 field-goal attempts." },
  { key: "efg_pct", label: "Shooting value (eFG%)", detail: "Shooting percentage adjusted for the extra value of three-pointers." },
  { key: "tov_pct", label: "Turnovers", detail: "The share of possessions that ended in a turnover." },
  { key: "ft_pct", label: "Free-throw possessions", detail: "The share of possessions that resulted in free throws, not free-throw shooting accuracy." },
];
const rates = [["fg_pct", "FG%"], ["efg_pct", "eFG%"], ["tov_pct", "Turnover possessions"],
  ["ft_pct", "Free-throw possessions"]] as const;
const count = (n: number | undefined) => n === undefined ? "—" : n.toLocaleString("en-US");
const percent = (n: number | null | undefined) => n == null ? "—" : `${(100 * n).toFixed(1)}%`;
const difference = (n: number | null) => {
  if (n === null) return "—";
  const rounded = Number(n.toFixed(1));
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)} pp`;
};
const sampleSize = (sample: LeagueSample | null, metric: LeagueRate) =>
  sample === null ? null : metric === "fg_pct" || metric === "efg_pct" ? sample.field_goal_attempts : sample.possessions;

function interpretation(metric: LeagueRate, side: "offense" | "defense", difference: number) {
  if (Math.abs(difference) < 0.05) return "About the same as the league";
  const higher = difference > 0;
  if (metric === "tov_pct") return side === "defense"
    ? higher ? "Forces more turnovers" : "Forces fewer turnovers"
    : higher ? "Turns the ball over more often" : "Turns the ball over less often";
  if (metric === "ft_pct") return side === "defense"
    ? higher ? "Allows more free-throw possessions" : "Allows fewer free-throw possessions"
    : higher ? "Gets free throws more often" : "Gets free throws less often";
  if (metric === "efg_pct") return side === "defense"
    ? higher ? "Allows more scoring value from shots" : "Allows less scoring value from shots"
    : higher ? "Gets more scoring value from shots" : "Gets less scoring value from shots";
  return side === "defense"
    ? higher ? "Opponents make shots more often" : "Opponents make shots less often"
    : higher ? "Makes shots more often" : "Makes shots less often";
}

function favorableDifference(metric: LeagueRate, side: "offense" | "defense", delta: number) {
  const higherBetter = (metric !== "tov_pct") === (side === "offense");
  return higherBetter ? delta : -delta;
}

/** Choose the headline from rows where both samples meet the selected rate’s cutoff. Keep smaller samples visible in the table. */
export function LeagueComparisonTable({ data, minimum = 20, initialMetric = "fg_pct" }: {
  data: LeagueComparison; minimum?: number; initialMetric?: LeagueRate;
}) {
  const titleId = useId();
  const metricId = useId();
  const [metric, setMetric] = useState<LeagueRate>(initialMetric);
  const { filters } = data;
  const selectedMetric = metrics.find(item => item.key === metric)!;
  const denominator = metric === "fg_pct" || metric === "efg_pct" ? "shot attempts" : "possessions";
  const eligible = data.rows.filter(row => row.difference_pp[metric] !== null
    && (sampleSize(row.team, metric) ?? 0) >= minimum && (sampleSize(row.league, metric) ?? 0) >= minimum);
  const standout = eligible.reduce<LeagueComparison["rows"][number] | null>((best, row) =>
    !best || Math.abs(row.difference_pp[metric]!) > Math.abs(best.difference_pp[metric]!) ? row : best, null);
  const standoutDelta = standout?.difference_pp[metric] ?? null;
  const barMaximum = metric === "efg_pct" ? 1.5 : 1;
  if (data.status === "no_data") return <section className={styles.card} role="status">
    <h2>League comparison</h2><p>No recorded {filters.side} data for {filters.team} in {filters.season}. Try another season.</p>
  </section>;
  return <section className={`${styles.card} ${readout.card}`} aria-labelledby={titleId}>
    <header className={readout.heading}><div><p className={readout.eyebrow}>League comparison</p>
      <h2 id={titleId}>How does {filters.team} compare?</h2>
      <p>{filters.season} · {filters.side === "defense" ? "Defense" : "Offense"} · Same play type, same season</p>
    </div><label className={readout.selector} htmlFor={metricId}>Compare
      <select id={metricId} value={metric} onChange={event => setMetric(event.target.value as LeagueRate)}>
        {metrics.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}
      </select>
    </label></header>
    <p className={readout.metricHelp}>{selectedMetric.detail} {filters.side === "defense"
      ? metric === "tov_pct" ? "On defense, more turnovers forced is favorable." : "On defense, lower opponent rates are favorable."
      : metric === "tov_pct" ? "Fewer turnovers is favorable." : "Higher rates are favorable."}</p>
    <div className={readout.headline} aria-live="polite">
      <span className={readout.eyebrow}>One thing to notice</span>
      {standout && standoutDelta !== null && Math.abs(standoutDelta) >= 0.05 ? <>
        <strong>{playtypeName(standout.play_type)}: {interpretation(metric, filters.side, standoutDelta).toLowerCase()}.</strong>
        <p>{filters.team} {percent(standout.team?.[metric])} · League {percent(standout.league?.[metric])}.
          {" "}{Math.abs(standoutDelta).toFixed(1)} percentage points {standoutDelta > 0 ? "above" : "below"} the league;
          the largest gap among rows with at least {count(minimum)} {denominator} on both sides.</p>
      </> : <><strong>{eligible.length ? "These rates are close to the league." : "More data is needed for a clear comparison."}</strong>
        <p>{eligible.length ? "All eligible differences round to less than 0.1 percentage point." : `No play type has a rate and at least ${count(minimum)} ${denominator} for both the team and league.`}</p></>}
    </div>
    <div className={`${readout.tableWrap} ${readout.mainTableWrap}`} tabIndex={0} role="region" aria-label="League comparison table">
      <table className={readout.readoutTable} role="table"><caption>{selectedMetric.label} by play type · bars start at zero</caption>
        <thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">Play type</th><th scope="col" role="columnheader">{filters.team}</th><th scope="col" role="columnheader">League</th><th scope="col" role="columnheader">What it shows</th></tr></thead>
        <tbody role="rowgroup">{data.rows.map(row => {
          const delta = row.difference_pp[metric];
          const small = [row.team, row.league].some(sample => sample !== null && (sampleSize(sample, metric) ?? 0) < minimum);
          const comparisonReady = delta !== null && !small;
          const tone = !comparisonReady || Math.abs(delta) < 0.05 ? "neutral" : favorableDifference(metric, filters.side, delta) > 0 ? "favorable" : "caution";
          return <tr key={row.play_type} role="row">
            <th scope="row" role="rowheader">{playtypeName(row.play_type)}{small ? <span className={readout.sample}>Small sample</span> : null}</th>
            {(["team", "league"] as const).map(side => <td key={side} role="cell" data-label={side === "team" ? filters.team : "League"}>
              <strong>{percent(row[side]?.[metric])}</strong>
              {row[side]?.[metric] == null ? <small>Unavailable</small> :
                <span className={readout.barTrack} aria-hidden="true"><span className={side === "team" ? readout.teamBar : readout.referenceBar}
                  style={{ width: `${100 * row[side]![metric]! / barMaximum}%` }} /></span>}
              <small>{count(sampleSize(row[side], metric) ?? undefined)} {denominator}</small>
            </td>)}
            <td className={readout.takeaway} role="cell" data-tone={tone}>{delta === null ? <span>Comparison unavailable</span> : <>
              <strong>{Math.abs(delta) < 0.05 ? "About the same" : `${Math.abs(delta).toFixed(1)} pp ${delta > 0 ? "above" : "below"} league`}</strong>
              <small>{small ? "Too few observations to highlight a difference." : interpretation(metric, filters.side, delta)}</small>
            </>}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p className={readout.footnote}>pp = percentage points. Small sample means fewer than {count(minimum)} {denominator}.
      {" "}The league includes {filters.team}. These describe recorded possessions, not the result of the next matchup.</p>
    <details className={`${styles.details} ${readout.details}`}><summary>Full numbers and calculations</summary>
      <div className={readout.tableWrap} tabIndex={0} role="region" aria-label="Full league comparison data">
        <table><caption>{filters.team} · {filters.season} regular season · {filters.side}</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Sample</th><th scope="col">Possessions</th><th scope="col">Attempts (FGA)</th>
            {rates.map(([key, label]) => <th key={key} scope="col">{label}</th>)}</tr></thead>
          {data.rows.map(row => <tbody key={row.play_type}>
            {(["team", "league"] as const).map((side, index) => <tr key={side}>
              {index === 0 ? <th scope="rowgroup" rowSpan={3}>{playtypeName(row.play_type)}</th> : null}
              <th scope="row">{side === "team" ? filters.team : "League"}</th>
              <td>{count(row[side]?.possessions)}</td><td>{count(row[side]?.field_goal_attempts)}</td>
              {rates.map(([key]) => <td key={key}>{percent(row[side]?.[key])}</td>)}
            </tr>)}
            <tr><th scope="row">Difference</th><td>—</td><td>—</td>
              {rates.map(([key]) => <td key={key}>{difference(row.difference_pp[key])}</td>)}</tr>
          </tbody>)}
        </table>
      </div>
      <p>Difference = {filters.team} minus league. FG% = total field goals made ÷ total field-goal attempts.
        eFG% weights each source eFG rate by its field-goal attempts; a made three counts 1.5 times a made two.</p>
      <p>Turnover and free-throw possession rates weight each source rate by possessions. Free-throw possessions is not free-throw shooting accuracy. These rates can overlap.</p>
      <p>eFG%, turnover and free-throw rates use rounded source rates. A zero denominator stays unavailable. League totals use the same season, side and play type, including the selected team.</p>
      <p>{filters.side === "defense" ? "Rates allowed by the defense are shown, except turnovers, which are forced by the defense. " : ""}Recorded regular-season data only; an ongoing season may be incomplete.
        Categories missing from both team and league data are not shown. No significance test is used to select the highlighted gap.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}
