import React from "react";
import { playtypeName, type UsageEfficiencyRow, type OpponentDefenseRow } from "../../../domain/playtype-dashboard";

export const signed = (value: number, digits = 3) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

export function UsageEfficiencyChart({ rows }: { rows: UsageEfficiencyRow[] }) {
  const xMax = Math.max(1, ...rows.map(r => Math.abs(r.usage_delta_pp))) * 1.2;
  const yMax = Math.max(0.05, ...rows.map(r => Math.abs(r.ppp_delta))) * 1.2;
  const x = (value: number) => 400 + value / xMax * 300;
  const y = (value: number) => 200 - value / yMax * 135;
  return <svg viewBox="0 0 800 410" role="img" aria-labelledby="usage-chart-title usage-chart-description">
    <title id="usage-chart-title">Usage and efficiency compared with the same league play type</title>
    <desc id="usage-chart-description">Right means more frequent; up means higher points per possession. Numbers match the rows below. Hollow circles mark small samples. This compares historical regular-season data.</desc>
    <rect x="100" y="65" width="300" height="135" fill="#f0fdfa" />
    <rect x="400" y="65" width="300" height="135" fill="#eff6ff" />
    <rect x="100" y="200" width="300" height="135" fill="#f8fafc" />
    <rect x="400" y="200" width="300" height="135" fill="#fff7ed" />
    <text x="112" y="49" fontSize="14" fill="#0f766e">Used less · higher PPP</text>
    <text x="688" y="49" textAnchor="end" fontSize="14" fill="#1d4ed8">Used more · higher PPP</text>
    <text x="112" y="357" fontSize="14" fill="#475569">Used less · lower PPP</text>
    <text x="688" y="357" textAnchor="end" fontSize="14" fill="#9a3412">Used more · lower PPP</text>
    {[-1, -0.5, 0, 0.5, 1].map(t => <g key={t}>
      <line x1={x(t * xMax)} x2={x(t * xMax)} y1="65" y2="335" stroke={t === 0 ? "#64748b" : "#e2e8f0"} strokeDasharray={t === 0 ? undefined : "3 4"} />
      <line x1="100" x2="700" y1={y(t * yMax)} y2={y(t * yMax)} stroke={t === 0 ? "#64748b" : "#e2e8f0"} strokeDasharray={t === 0 ? undefined : "3 4"} />
      <text x="88" y={y(t * yMax) + 4} textAnchor="end" fontSize="12" fill="#475569">{signed(t * yMax, 2)}</text>
    </g>)}
    <text x="100" y="382" fontSize="12" fill="#475569">{signed(-xMax, 1)} pp</text>
    <text x="400" y="382" textAnchor="middle" fontSize="12" fill="#475569">0</text>
    <text x="700" y="382" textAnchor="end" fontSize="12" fill="#475569">{signed(xMax, 1)} pp</text>
    <text x="400" y="405" textAnchor="middle" fontSize="14" fill="#334155">Usage difference from league (percentage points)</text>
    <text x="400" y="20" textAnchor="middle" fontSize="14" fill="#334155">PPP difference from league</text>
    {rows.map((r, i) => <g key={r.play_type}>
      <title>{playtypeName(r.play_type)}: usage {r.usage_pct.toFixed(1)}%, PPP {r.ppp.toFixed(3)}; {r.possessions} possessions{r.sample_ok ? "" : "; small sample"}</title>
      <circle cx={x(r.usage_delta_pp)} cy={y(r.ppp_delta)} r="12" fill={r.sample_ok ? "#5b21b6" : "white"} stroke="#5b21b6" strokeWidth="2" />
      <text x={x(r.usage_delta_pp)} y={y(r.ppp_delta) + 4} textAnchor="middle" fontSize="11" fontWeight="700" fill={r.sample_ok ? "white" : "#5b21b6"}>{i + 1}</text>
    </g>)}
  </svg>;
}

export function OpponentDefenseChart({ rows }: { rows: OpponentDefenseRow[] }) {
  const extent = Math.max(0.03, ...rows.map(r => Math.abs(r.allowed_ppp_delta))) * 1.3;
  const center = 452, scale = 205 / extent, height = 70 + rows.length * 48;
  return <svg viewBox={`0 0 800 ${height}`} role="img" aria-labelledby="defense-chart-title defense-chart-description">
    <title id="defense-chart-title">Opponent allowed PPP relative to league, sorted from highest to lowest difference</title>
    <desc id="defense-chart-description">Bars to the right mean more points allowed than the league for the same play type. Dashed bars mark small samples. Exact values and possessions appear in the table.</desc>
    <text x="235" y="20" fontSize="13" fill="#0f766e">Allows less than league</text>
    <text x="675" y="20" textAnchor="end" fontSize="13" fill="#9a3412">Allows more than league</text>
    <line x1={center} x2={center} y1="29" y2={height - 25} stroke="#64748b" />
    {rows.map((r, i) => {
      const y = 44 + i * 48, width = Math.abs(r.allowed_ppp_delta) * scale;
      return <g key={r.play_type}>
        <title>{playtypeName(r.play_type)}: {r.allowed_ppp.toFixed(3)} allowed PPP; league {r.league_allowed_ppp.toFixed(3)}; {r.possessions} possessions{r.sample_ok ? "" : "; small sample"}</title>
        <text x="12" y={y + 17} fontSize="15" fill="#1e293b">{playtypeName(r.play_type)}</text>
        <rect x={Math.min(center, center + r.allowed_ppp_delta * scale)} y={y} width={Math.max(1, width)} height="27" rx="4"
          fill={r.sample_ok ? (r.allowed_ppp_delta > 0 ? "#c2410c" : "#0f766e") : "#f1f5f9"}
          stroke={r.allowed_ppp_delta > 0 ? "#9a3412" : "#0f766e"} strokeDasharray={r.sample_ok ? undefined : "4 3"} />
        <text x="716" y={y + 17} fontSize="14" fill="#334155">{signed(r.allowed_ppp_delta)}</text>
      </g>;
    })}
    <text x={center} y={height - 5} textAnchor="middle" fontSize="12" fill="#475569">0 · league average for each play type</text>
  </svg>;
}
