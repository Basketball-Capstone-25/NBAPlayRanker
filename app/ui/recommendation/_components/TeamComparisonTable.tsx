"use client";

import React, { useId, useState } from "react";
import type { TeamComparison, TeamProfile } from "../../../domain/team-comparison";
import { playtypeName } from "../../../domain/playtype-dashboard";
import styles from "./matchup-dashboard.module.css";
import readout from "./comparison-readouts.module.css";

type ProfileMetric = "ppp" | "usage";
const count = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString("en-US");
const rate = (value: number | null | undefined, digits: number, unit = "") =>
  value == null ? "—" : `${value.toFixed(digits)}${unit}`;
const difference = (value: number | null, digits: number, unit = "") => {
  if (value === null) return "—";
  const rounded = Number(value.toFixed(digits));
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(digits)}${unit}`;
};
const displayValue = (sample: TeamProfile | null, metric: ProfileMetric) =>
  metric === "usage" ? sample?.usage_pct ?? null : sample?.ppp == null ? null : sample.ppp * 100;
const displayDifference = (row: TeamComparison["rows"][number], metric: ProfileMetric) =>
  metric === "usage" ? row.difference.usage_pp : row.difference.ppp === null ? null : row.difference.ppp * 100;

function observation(data: TeamComparison, row: TeamComparison["rows"][number], metric: ProfileMetric) {
  const delta = displayDifference(row, metric);
  if (delta === null) return "Comparison unavailable";
  if (Math.abs(delta) < 0.05) return metric === "usage" ? "Used about as often" : "About the same scoring rate";
  const higherTeam = delta > 0 ? data.filters.team : data.filters.opponent;
  return metric === "usage" ? `${higherTeam} used this play more often`
    : `${higherTeam} scored ${Math.abs(delta).toFixed(1)} more points per 100 of these possessions`;
}

export function TeamComparisonTable({ data, minimum = 20, initialMetric = "ppp" }: {
  data: TeamComparison; minimum?: number; initialMetric?: ProfileMetric;
}) {
  const titleId = useId();
  const metricId = useId();
  const [metric, setMetric] = useState<ProfileMetric>(initialMetric);
  const { filters } = data;
  const missingTeams = [data.team_possessions === null ? filters.team : null,
    data.opponent_possessions === null ? filters.opponent : null].filter(Boolean);
  const eligible = data.rows.filter(row => displayDifference(row, metric) !== null
    && (row.team?.possessions ?? 0) >= minimum && (row.opponent?.possessions ?? 0) >= minimum);
  const standout = eligible.reduce<TeamComparison["rows"][number] | null>((best, row) =>
    !best || Math.abs(displayDifference(row, metric)!) > Math.abs(displayDifference(best, metric)!) ? row : best, null);
  const standoutDelta = standout ? displayDifference(standout, metric) : null;
  const barMaximum = metric === "usage" ? 100 : Math.max(100, ...data.rows.flatMap(row =>
    [displayValue(row.team, metric) ?? 0, displayValue(row.opponent, metric) ?? 0]));
  return <section className={`${styles.card} ${readout.card}`} aria-labelledby={titleId}>
    <header className={readout.heading}><div><p className={readout.eyebrow}>Team profiles</p>
      <h2 id={titleId}>{filters.team} and {filters.opponent}: how they play</h2>
      <p>{filters.season} · Recorded regular-season offense</p>
    </div><label className={readout.selector} htmlFor={metricId}>Compare
      <select id={metricId} value={metric} onChange={event => setMetric(event.target.value as ProfileMetric)}>
        <option value="ppp">Scoring rate</option><option value="usage">Play mix</option>
      </select>
    </label></header>
    <p className={readout.metricHelp}>{metric === "ppp"
      ? "Points per 100 possessions of each play type. This puts the two teams on the same scale even when they recorded different numbers of possessions."
      : "How often each team used a play, as a share of its recorded play-type possessions. This shows style, not which team is better."}</p>
    {missingTeams.length ? <p className={readout.notice} role="status">No recorded offensive data for {missingTeams.join(" or ")} in {filters.season}.</p> : null}
    {data.rows.length ? <>
      <div className={readout.headline} aria-live="polite"><span className={readout.eyebrow}>One thing to notice</span>
        {standout && standoutDelta !== null && Math.abs(standoutDelta) >= 0.05 ? <>
          <strong>{playtypeName(standout.play_type)}: {observation(data, standout, metric)}.</strong>
          <p>{metric === "usage" ? `${Math.abs(standoutDelta).toFixed(1)} percentage points apart. ` : ""}
            The largest {metric === "usage" ? "play-mix" : "scoring-rate"} gap among categories with at least {count(minimum)} recorded possessions for each team.
            {" "}{metric === "usage" ? "Use this to compare what each team tends to run." : "Use this to choose a play type to review on film, not to predict extra points."}</p>
        </> : <><strong>{eligible.length ? "The two profiles are close on this measure." : "More shared data is needed for a clear comparison."}</strong>
          <p>{eligible.length ? "All eligible differences round to less than 0.1 on this scale." : `No play type has an available rate and at least ${count(minimum)} recorded possessions for both teams.`}</p></>}
      </div>
      <div className={`${readout.tableWrap} ${readout.mainTableWrap}`} tabIndex={0} role="region" aria-label="Team profile comparison table">
        <table className={readout.readoutTable} role="table"><caption>{metric === "usage" ? "Share of recorded possessions (%)" : "Points per 100 play-type possessions"} · bars start at zero</caption>
          <thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">Play type</th><th scope="col" role="columnheader">{filters.team}</th><th scope="col" role="columnheader">{filters.opponent}</th><th scope="col" role="columnheader">What it shows</th></tr></thead>
          <tbody role="rowgroup">{data.rows.map(row => {
            const delta = displayDifference(row, metric);
            const small = [row.team, row.opponent].some(sample => sample !== null && sample.possessions < minimum);
            return <tr key={row.play_type} role="row">
              <th scope="row" role="rowheader">{playtypeName(row.play_type)}{small ? <span className={readout.sample}>Small sample</span> : null}</th>
              {(["team", "opponent"] as const).map(side => {
                const value = displayValue(row[side], metric);
                return <td key={side} role="cell" data-label={filters[side]}><strong>{rate(value, 1, metric === "usage" ? "%" : "")}</strong>
                  {value === null ? <small>Unavailable</small> :
                    <span className={readout.barTrack} aria-hidden="true"><span className={side === "team" ? readout.teamBar : readout.referenceBar}
                      style={{ width: `${100 * value / barMaximum}%` }} /></span>}
                  <small>{count(row[side]?.possessions)} possessions</small>
                </td>;
              })}
              <td className={readout.takeaway} role="cell" data-tone="neutral">{delta === null ? <span>Comparison unavailable</span> : <>
                <strong>{metric === "usage" ? Math.abs(delta) < 0.05 ? "About the same share" : `${Math.abs(delta).toFixed(1)} pp apart`
                  : Math.abs(delta) < 0.05 ? "About the same rate" : `${Math.abs(delta).toFixed(1)} points apart`}</strong>
                <small>{small ? "Too few possessions to highlight a difference." : observation(data, row, metric)}</small>
              </>}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className={readout.footnote}>These are separate team profiles, not results from games against each other or a forecast.
        {" "}{metric === "ppp" ? "A higher play-type rate does not mean running more of that play will raise team scoring." : "pp = percentage points. A more common play is not necessarily a better play."}
        {" "}Small sample: fewer than {count(minimum)} recorded possessions.</p>
    </> : null}
    <details className={`${styles.details} ${readout.details}`}><summary>Full numbers and calculations</summary>
      <p>{filters.team} and {filters.opponent} season profiles. Recorded possessions: {filters.team} {count(data.team_possessions)} · {filters.opponent} {count(data.opponent_possessions)}.</p>
      {data.rows.length ? <div className={readout.tableWrap} tabIndex={0} role="region" aria-label="Full team profile data">
        <table><caption>{filters.team} and {filters.opponent} · {filters.season} regular season · offense</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Team</th><th scope="col">Possessions</th>
            <th scope="col">Usage</th><th scope="col">PPP</th><th scope="col">Points</th></tr></thead>
          {data.rows.map(row => <tbody key={row.play_type}>
            {(["team", "opponent"] as const).map((side, index) => <tr key={side}>
              {index === 0 ? <th scope="rowgroup" rowSpan={3}>{playtypeName(row.play_type)}</th> : null}
              <th scope="row">{filters[side]}</th><td>{count(row[side]?.possessions)}</td>
              <td>{rate(row[side]?.usage_pct, 1, "%")}</td><td>{rate(row[side]?.ppp, 3)}</td><td>{count(row[side]?.points)}</td>
            </tr>)}
            <tr><th scope="row">Difference</th><td>—</td>
              <td>{difference(row.difference.usage_pp, 1, " pp")}</td><td>{difference(row.difference.ppp, 3)}</td><td>—</td></tr>
          </tbody>)}
        </table>
      </div> : null}
      <p>Usage = play-type possessions ÷ that team's total recorded play-type possessions × 100.
        Each team uses its own total, including categories missing from the other team. Only qualifying source rows are included; this is not a count of every team possession.</p>
      <p>PPP = points ÷ possessions. The scoring-rate view multiplies PPP by 100; it is not whole-team offensive rating.
        Difference = {filters.team} minus {filters.opponent}. Differences subtract the unrounded values.
        A zero denominator stays unavailable; a recorded zero is shown as 0.</p>
      <p>Categories recorded for either team are included. Missing categories stay unavailable. An ongoing season may be incomplete.
        The highlighted gap is descriptive; no significance test or matchup prediction is used.</p>
      <p>Source: <code>{data.dataset_version}</code><br />Aggregation: <code>{data.aggregation_version}</code></p>
    </details>
  </section>;
}
