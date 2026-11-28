// What-if control for moving possessions between two play types.

"use client";

import React, { useState } from "react";
import { playtypeName, type UsageEfficiencyRow } from "../../../domain/playtype-dashboard";
import { possessionScenario } from "../../../domain/possession-scenario";
import styles from "./coach-reading.module.css";

/** Limit the move to the source play’s recorded share. The estimate holds both historical scoring rates fixed. */
export default function PossessionScenario({ rows }: { rows: UsageEfficiencyRow[] }) {
  const supported = rows.filter(row => row.sample_ok && row.possessions > 0);
  const [fromKey, setFrom] = useState(""), [toKey, setTo] = useState("");
  const [amount, setAmount] = useState(1);
  const from = supported.find(row => row.play_type === fromKey), to = supported.find(row => row.play_type === toKey);
  const maximum = from ? Math.floor(from.usage_pct) : 0;
  const count = Math.min(amount, maximum);
  const result = from && to ? possessionScenario(from, to, count) : null;
  const displayedDifference = result && Math.abs(result.difference) >= .05 ? result.difference : 0;
  return <details className={styles.readout}>
    <summary><strong>What if we changed the play mix?</strong></summary>
    <p>Move a few of every 100 recorded offensive possessions from one play to another. See how much the scoring rates differ.</p>
    {supported.length < 2 ? <p>Two play types must meet the sample cutoff to compare them here.</p> : <>
      <div className={styles.controls}>
        <label className={styles.label}>Use less<select value={from?.play_type ?? ""} onChange={event => setFrom(event.target.value)}>
          <option value="">Choose a play</option>{supported.filter(row => row.usage_pct >= 1).map(row => <option value={row.play_type} key={row.play_type}>{playtypeName(row.play_type)}</option>)}
        </select></label>
        <label className={styles.label}>Use more<select value={to?.play_type ?? ""} onChange={event => setTo(event.target.value)}>
          <option value="">Choose a play</option>{supported.map(row => <option value={row.play_type} key={row.play_type}>{playtypeName(row.play_type)}</option>)}
        </select></label>
      </div>
      {maximum > 0 ? <label className={styles.label}>Move {count} of every 100 recorded possessions
        <input type="range" min={1} max={maximum} value={count} onChange={event => setAmount(Number(event.target.value))} />
      </label> : null}
      {from && to && from.play_type === to.play_type ? <p>Choose two different plays.</p> : null}
      {result && from && to ? <output className={styles.result} aria-live="polite">
        <strong>{displayedDifference > 0 ? "+" : ""}{displayedDifference.toFixed(1)} points</strong>
        per 100 recorded offensive possessions, if both scoring rates stayed the same.
        <p className={styles.quiet}>{count} × ({to.ppp.toFixed(3)} − {from.ppp.toFixed(3)}) = {result.difference.toFixed(2)} points. Calculated before rounding.</p>
      </output> : null}
      <p className={styles.quiet}>This is a fixed-rate illustration, not a prediction or points per game. More calls can mean tougher shots, different defenders or fewer chances for another play. Use it to choose something to test in practice.</p>
    </>}
  </details>;
}
