// Per-play scoring, turnover and foul rates with their source coverage.

import React, { useId, useState } from "react";
import { OUTCOME_METRICS, type OutcomeRate, type PossessionOutcomes } from "../../../domain/possession-outcomes";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";
import reading from "./coach-reading.module.css";

const count = (value: number) => value.toLocaleString("en-US");
const display = (rate: OutcomeRate) => rate.status === "invalid" ? "Invalid source"
  : rate.value === null ? "—" : `${(100 * rate.value).toFixed(1)}%`;
const coverage = (rate: OutcomeRate, possessions: number) =>
  `${count(rate.usable_possessions)} of ${count(possessions)} possessions have a usable rate; `
  + `${count(rate.missing_possessions)} missing; ${count(rate.invalid_possessions)} invalid.`;

/** Give each outcome its own scale because a possession can count toward multiple rates. The detail rows show how much usable data each rate has. */
export function PossessionOutcomesTable({ data, minimum = 20 }: { data: PossessionOutcomes; minimum?: number }) {
  const titleId = useId();
  const { filters } = data;
  const [selected, setSelected] = useState("");
  const focus = data.rows.find(row => row.play_type === selected)
    ?? [...data.rows].sort((a, b) => b.possessions - a.possessions)[0];
  return <section className={styles.card} aria-labelledby={titleId}>
    <div className={styles.sectionHeading}><div><h2 id={titleId}>{filters.team} possession outcomes</h2>
      <p>Rates can overlap. Do not add these percentages.</p>
    </div></div>
    {data.rows.length ? <>
      {focus ? <div className={reading.readout}>
        <h3>What happens when this play ends?</h3>
        <label className={reading.label}>Play type<select value={focus.play_type} onChange={event => setSelected(event.target.value)}>
          {data.rows.map(row => <option value={row.play_type} key={row.play_type}>{playtypeName(row.play_type)}</option>)}
        </select></label>
        <p>{focus.possessions < minimum ? "Small sample — treat these rates cautiously." : `${count(focus.possessions)} recorded possessions.`}
          {focus.rates.scoring.value !== null ? ` About ${(focus.rates.scoring.value * 100).toFixed(1)} out of every 100 produced points.` : " The scoring rate is unavailable."}
          {focus.rates.turnover.value !== null ? ` About ${(focus.rates.turnover.value * 100).toFixed(1)} ended in a turnover.` : " The turnover rate is unavailable."}</p>
        <div className={reading.bars}>{OUTCOME_METRICS.map(metric => {
          const rate = focus.rates[metric.key];
          return <div key={metric.key}><div className={reading.barLabel}><span>{metric.label}</span><strong>{display(rate)}</strong></div>
            {rate.value !== null ? <div className={reading.track}><span className={reading.fill} style={{ width: `${rate.value * 100}%` }} /></div> : null}
          </div>;
        })}</div>
        <p className={reading.quiet}>Check scoring alongside turnovers before relying on a play. Each bar has its own 0–100% scale; a scoring possession can also include a foul or an and-one.</p>
      </div> : null}
      <p className={styles.note}>Approximate rates from rounded source data. — means unavailable.</p>
      <details className={styles.details}><summary>Compare all play types and source coverage</summary>
      <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Possession outcome rates table">
        <table><caption>{filters.team} · {filters.season} regular season · offense</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Possessions</th>
            {OUTCOME_METRICS.map(metric => <th scope="col" key={metric.key}>{metric.label}</th>)}
          </tr></thead>
          <tbody>{data.rows.map(row => <tr key={row.play_type}>
            <th scope="row">{playtypeName(row.play_type)}{row.possessions < minimum
              ? <> <span className={styles.smallSample}>Small sample</span></> : null}</th>
            <td>{count(row.possessions)}</td>
            {OUTCOME_METRICS.map(metric => {
              const rate = row.rates[metric.key], text = display(rate), description = coverage(rate, row.possessions);
              return <td key={metric.key} title={description} aria-label={`${metric.label}: ${text}. ${description}`}>{text}</td>;
            })}
          </tr>)}</tbody>
        </table>
      </div>
      </details>
      <p className={styles.note}>Small sample: fewer than {count(minimum)} possessions. The cutoff flags rows; it does not remove or change them.</p>
    </> : <p role="status">No recorded offensive data for {filters.team} in {filters.season}.</p>}
    <details className={styles.details}><summary>Rate definitions and calculations</summary>
      <dl>{OUTCOME_METRICS.map(metric => <React.Fragment key={metric.key}>
        <dt><strong>{metric.label}</strong></dt><dd>{metric.definition} <code>{metric.source}</code></dd>
      </React.Fragment>)}</dl>
      <p>Rate = Σ(player possessions × source rate) ÷ total play-type possessions. Displayed percentage = rate × 100.</p>
      <p>Each rate needs valid values for all recorded possessions. A missing or invalid source value leaves that rate unavailable; the other rates can still be shown. Zero possessions have no rate.</p>
      <p>The source rates are rounded, so these percentages are approximate. They do not recover exact event counts or form a 100% distribution.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}
