import React, { useId } from "react";
import type { TeamComparison } from "../../../domain/team-comparison";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";

const count = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString("en-US");
const rate = (value: number | null | undefined, digits: number, unit = "") =>
  value == null ? "—" : `${value.toFixed(digits)}${unit}`;
const difference = (value: number | null, digits: number, unit = "") => {
  if (value === null) return "—";
  const rounded = Number(value.toFixed(digits));
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(digits)}${unit}`;
};

export function TeamComparisonTable({ data, minimum = 20 }: { data: TeamComparison; minimum?: number }) {
  const titleId = useId();
  const { filters } = data;
  const missingTeams = [data.team_possessions === null ? filters.team : null,
    data.opponent_possessions === null ? filters.opponent : null].filter(Boolean);
  return <section className={styles.card} aria-labelledby={titleId}>
    <div className={styles.sectionHeading}><div><h2 id={titleId}>{filters.team} and {filters.opponent} season profiles</h2>
      <p>Regular-season offense. These are season-long profiles, not results from games against each other or a forecast.</p>
    </div></div>
    <p>Full-season possessions: {filters.team} {count(data.team_possessions)} · {filters.opponent} {count(data.opponent_possessions)}.</p>
    {missingTeams.length ? <p role="status">No recorded offensive data for {missingTeams.join(" or ")} in {filters.season}.</p> : null}
    {data.rows.length ? <>
      <p className={styles.note}>Difference = {filters.team} minus {filters.opponent}. Usage differences are percentage points (pp). — means unavailable.</p>
      <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Team profile comparison table">
        <table><caption>{filters.team} and {filters.opponent} · {filters.season} regular season · offense</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Team</th><th scope="col">Possessions</th>
            <th scope="col">Usage</th><th scope="col">PPP</th></tr></thead>
          {data.rows.map(row => <tbody key={row.play_type}>
            {(["team", "opponent"] as const).map((side, index) => {
              const sample = row[side];
              return <tr key={side}>
                {index === 0 ? <th scope="rowgroup" rowSpan={3}>{playtypeName(row.play_type)}</th> : null}
                <th scope="row">{filters[side]}{sample && sample.possessions < minimum
                  ? <> <span className={styles.smallSample}>Small sample</span></> : null}</th>
                <td>{count(sample?.possessions)}</td><td>{rate(sample?.usage_pct, 1, "%")}</td><td>{rate(sample?.ppp, 3)}</td>
              </tr>;
            })}
            <tr><th scope="row">Difference</th><td>—</td>
              <td>{difference(row.difference.usage_pp, 1, " pp")}</td><td>{difference(row.difference.ppp, 3)}</td></tr>
          </tbody>)}
        </table>
      </div>
      <p className={styles.note}>Small sample: fewer than {count(minimum)} possessions. The cutoff flags rows; it does not remove or change them.</p>
    </> : null}
    <details className={styles.details}><summary>How these profiles are calculated</summary>
      <p>Usage = play-type possessions ÷ that team's full-season possessions × 100. Each team uses its own total, including categories missing from the other team.</p>
      <p>PPP = points ÷ possessions. Differences subtract the unrounded values. A zero denominator stays unavailable; a recorded zero is shown as 0.</p>
      <p>Categories recorded for either team are included. Missing categories stay unavailable. These profiles describe the selected season; they do not predict a matchup result.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}
