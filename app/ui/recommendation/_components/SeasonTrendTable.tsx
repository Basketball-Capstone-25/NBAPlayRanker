import React from "react";
import type { SeasonTrend } from "../../../domain/season-trend";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";

const count = (value: number | null) => value === null ? "—" : value.toLocaleString("en-US");

export function SeasonTrendTable({ data, minimum }: { data: SeasonTrend; minimum: number }) {
  return <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Season trend table">
    <table>
      <caption>{data.filters.team} · {playtypeName(data.filters.play_type)} · offensive regular-season totals</caption>
      <thead><tr><th scope="col">Season</th><th scope="col">Play possessions</th><th scope="col">Points</th>
        <th scope="col">Team possessions</th><th scope="col">Usage</th><th scope="col">PPP</th><th scope="col">Sample</th></tr></thead>
      <tbody>{data.rows.map(row => <tr key={row.season}>
        <th scope="row">{row.season}</th><td>{count(row.possessions)}</td><td>{count(row.points)}</td>
        <td>{count(row.team_possessions)}</td><td>{row.usage_pct === null ? "—" : `${row.usage_pct.toFixed(1)}%`}</td>
        <td>{row.ppp === null ? "—" : row.ppp.toFixed(3)}</td>
        <td>{row.possessions === null ? "Not recorded" : row.possessions < minimum
          ? <span className={styles.smallSample}>Small sample</span> : "Recorded"}</td>
      </tr>)}</tbody>
    </table>
  </div>;
}
