import React from "react";
import type { SeasonTrend } from "../../../domain/season-trend";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";

export type TrendMetric = "ppp" | "usage_pct";

export function SeasonTrendChart({ data, metric, minimum }: { data: SeasonTrend; metric: TrendMetric; minimum: number }) {
  const { rows, filters } = data;
  const label = metric === "ppp" ? "PPP" : "Usage (%)";
  const maximum = metric === "usage_pct"
    ? Math.min(100, Math.max(5, Math.ceil(Math.max(...rows.map(row => row.usage_pct ?? 0)) / 5) * 5))
    : Math.max(1, Math.ceil(Math.max(...rows.map(row => row.ppp ?? 0)) * 4) / 4);
  const firstYear = Number(rows[0].season.slice(0, 4));
  const span = Math.max(1, Number(rows[rows.length - 1].season.slice(0, 4)) - firstYear);
  const x = (season: string) => rows.length === 1 ? 400 : 60 + 680 * (Number(season.slice(0, 4)) - firstYear) / span;
  const y = (value: number) => 230 - 180 * value / maximum;
  let connected = false;
  const path = rows.map(row => {
    const value = row[metric];
    if (value === null) { connected = false; return ""; }
    const command = `${connected ? "L" : "M"} ${x(row.season)} ${y(value)}`;
    connected = true;
    return command;
  }).filter(Boolean).join(" ");
  const title = `${filters.team} ${playtypeName(filters.play_type)}: ${label} across regular seasons`;
  return <div className={styles.chart}>
    <svg viewBox="0 0 780 290" role="img" aria-label={title}>
      <title>{title}</title>
      <desc>One point per recorded season. Lines stop at missing values. Hollow circles mark small samples. Exact values are in the table.</desc>
      <text x="60" y="24" fontSize="13" fill="#334155">{label}</text>
      {[0, 1, 2, 3, 4].map(tick => {
        const value = maximum * tick / 4;
        return <g key={tick}><line x1="60" x2="740" y1={y(value)} y2={y(value)} stroke="#e2e8f0" />
          <text x="50" y={y(value) + 4} textAnchor="end" fontSize="11" fill="#475569">
            {metric === "usage_pct" ? `${Number(value.toFixed(2))}%` : value.toFixed(3)}</text></g>;
      })}
      <path d={path} fill="none" stroke="#5b21b6" strokeWidth="2" />
      {rows.map(row => <g key={row.season}>
        <text x={x(row.season)} y="258" textAnchor="middle" fontSize="11" fill="#475569">{row.season}</text>
        {row[metric] === null ? <text x={x(row.season)} y="279" textAnchor="middle" fontSize="10" fill="#64748b">No data</text>
          : <circle cx={x(row.season)} cy={y(row[metric])} r="5" stroke="#5b21b6" strokeWidth="2"
            fill={(row.possessions ?? 0) < minimum ? "white" : "#5b21b6"}>
            <title>{`${row.season}: ${row[metric].toFixed(metric === "ppp" ? 3 : 1)}${metric === "ppp" ? " PPP" : "% usage"}; ${row.possessions} possessions`}</title>
          </circle>}
      </g>)}
    </svg>
  </div>;
}
