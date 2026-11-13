import React from "react";
import type { LeagueComparison } from "../../../domain/league-comparison";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";

const rates = [["fg_pct", "FG%"], ["efg_pct", "eFG%"], ["tov_pct", "Turnover possessions"],
  ["ft_pct", "Free-throw possessions"]] as const;
const count = (n: number | undefined) => n === undefined ? "—" : n.toLocaleString("en-US");
const percent = (n: number | null | undefined) => n == null ? "—" : `${(100 * n).toFixed(1)}%`;
const difference = (n: number | null) => {
  if (n === null) return "—";
  const rounded = Number(n.toFixed(1));
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)} pp`;
};

export function LeagueComparisonTable({ data, minimum = 20 }: { data: LeagueComparison; minimum?: number }) {
  const { filters } = data;
  if (data.status === "no_data") return <section className={styles.card} role="status">
    <h2>League comparison</h2><p>No recorded {filters.side} data for {filters.team} in {filters.season}. Try another season.</p>
  </section>;
  return <section className={styles.card} aria-labelledby="league-title">
    <div className={styles.sectionHeading}><div><h2 id="league-title">{filters.team} compared with the league</h2>
      <p>{filters.side === "defense" ? "Rates allowed by the defense." : "Recorded offensive rates."} The league includes {filters.team}.</p>
    </div></div>
    <p className={styles.note}>Each difference is team minus league in percentage points (pp). Higher is not always better. — means unavailable.</p>
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="League comparison table">
      <table><caption>{filters.team} · {filters.season} regular season · {filters.side}</caption>
        <thead><tr><th scope="col">Play type</th><th scope="col">Sample</th><th scope="col">Possessions</th><th scope="col">Attempts (FGA)</th>
          {rates.map(([key, label]) => <th key={key} scope="col">{label}</th>)}</tr></thead>
        {data.rows.map(row => <tbody key={row.play_type}>
          {(["team", "league"] as const).map((side, index) => {
            const sample = row[side];
            return <tr key={side}>
              {index === 0 ? <th scope="rowgroup" rowSpan={3}>{playtypeName(row.play_type)}</th> : null}
              <th scope="row">{side === "team" ? filters.team : "League"}
                {sample && (sample.possessions < minimum || sample.field_goal_attempts < minimum)
                  ? <> <span className={styles.smallSample}>Small sample</span></> : null}</th>
              <td>{count(sample?.possessions)}</td><td>{count(sample?.field_goal_attempts)}</td>
              {rates.map(([key]) => <td key={key}>{percent(sample?.[key])}</td>)}
            </tr>;
          })}
          <tr><th scope="row">Difference</th><td>—</td><td>—</td>
            {rates.map(([key]) => <td key={key}>{difference(row.difference_pp[key])}</td>)}</tr>
        </tbody>)}
      </table>
    </div>
    <p className={styles.note}>Small sample: fewer than {count(minimum)} possessions or field-goal attempts. The cutoff flags rows; it does not remove or change them.</p>
    <details className={styles.details}><summary>How these rates are calculated</summary>
      <p>FG% = total field goals made ÷ total field-goal attempts. eFG% weights each source eFG rate by its field-goal attempts.</p>
      <p>Turnover and free-throw possession rates weight each source rate by possessions. Free-throw possessions is not free-throw shooting accuracy. These rates can overlap.</p>
      <p>eFG%, turnover and free-throw rates use rounded source rates. A zero denominator stays unavailable. League totals use the same season, side and play type, including the selected team.</p>
      <p>Historical regular-season data only. Categories missing from both team and league data are not shown. These are not forecasts or results of games against one opponent.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}
