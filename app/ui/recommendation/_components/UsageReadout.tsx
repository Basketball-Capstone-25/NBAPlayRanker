"use client";

import React, { useState } from "react";
import { playtypeName, type PlaytypeDashboard } from "../../../domain/playtype-dashboard";
import styles from "./coach-reading.module.css";

export default function UsageReadout({ data }: { data: PlaytypeDashboard }) {
  const rows = data.usage_efficiency.rows;
  const [selected, setSelected] = useState("");
  const suggested = [...rows].filter(row => row.sample_ok).sort((a, b) => b.ppp_delta - a.ppp_delta)[0];
  const row = rows.find(item => item.play_type === selected) ?? suggested ?? rows[0];
  if (!row) return null;
  const defense = data.opponent_defense.rows.find(item => item.play_type === row.play_type);
  const difference = row.ppp_delta * 100;
  const close = Math.abs(difference) < .05;
  return <div className={styles.readout}>
    <label className={styles.label}>Take a closer look at<select value={row.play_type} onChange={event => setSelected(event.target.value)}>
      {rows.map((item, index) => <option value={item.play_type} key={item.play_type}>{index + 1} · {playtypeName(item.play_type)}</option>)}
    </select></label>
    <div className={styles.numbers}>
      <span><strong>{(row.ppp * 100).toFixed(1)}</strong>{data.filters.our} points per 100 of these possessions</span>
      <span><strong>{(row.league_ppp * 100).toFixed(1)}</strong>league points per 100 of these possessions</span>
      <span><strong>{row.usage_pct.toFixed(1)}%</strong>of {data.filters.our}’s recorded play mix</span>
    </div>
    <h3>{!row.sample_ok ? "Too few possessions for a strong takeaway" : close ? "Scoring is close to the league rate" : `${Math.abs(difference).toFixed(1)} ${difference > 0 ? "more" : "fewer"} points per 100 of these possessions`}</h3>
    <p>{data.filters.our} used {playtypeName(row.play_type).toLowerCase()} on {row.usage_pct.toFixed(1)}% of recorded possessions; the league used it on {row.league_usage_pct.toFixed(1)}%.</p>
    {defense?.sample_ok && row.sample_ok ? <p>{data.filters.opp} allowed {(defense.allowed_ppp * 100).toFixed(1)} points per 100 of these possessions, compared with {(defense.league_allowed_ppp * 100).toFixed(1)} for the league.
      {row.ppp_delta > .0005 && defense.allowed_ppp_delta > .0005 ? " Our scoring and their defense both make this worth reviewing on film." : " Check how their defenders handle this action before changing the game plan."}</p>
      : <p>{!row.sample_ok ? "Review the recorded plays before drawing a conclusion from this sample." : defense ? "The opponent’s sample is too small for a supported comparison." : "This source has no opponent defensive rate for this play."}</p>}
    <p className={styles.quiet}>These are separate season profiles. They show what happened, not what will happen when these teams play.</p>
  </div>;
}
