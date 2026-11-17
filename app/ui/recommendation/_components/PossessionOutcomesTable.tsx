import React, { useId } from "react";
import { OUTCOME_METRICS, type OutcomeRate, type PossessionOutcomes } from "../../../domain/possession-outcomes";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";

const count = (value: number) => value.toLocaleString("en-US");
const display = (rate: OutcomeRate) => rate.status === "invalid" ? "Invalid source"
  : rate.value === null ? "—" : `${(100 * rate.value).toFixed(1)}%`;
const coverage = (rate: OutcomeRate, possessions: number) =>
  `${count(rate.usable_possessions)} of ${count(possessions)} possessions have a usable rate; `
  + `${count(rate.missing_possessions)} missing; ${count(rate.invalid_possessions)} invalid.`;

export function PossessionOutcomesTable({ data, minimum = 20 }: { data: PossessionOutcomes; minimum?: number }) {
  const titleId = useId();
  const { filters } = data;
  return <section className={styles.card} aria-labelledby={titleId}>
    <div className={styles.sectionHeading}><div><h2 id={titleId}>{filters.team} possession outcomes</h2>
      <p>Rates can overlap. Do not add these percentages.</p>
    </div></div>
    {data.rows.length ? <>
      <p className={styles.note}>Approximate rates from rounded source data. — means unavailable.</p>
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
