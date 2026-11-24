"use client";

import { useEffect, useState } from "react";
import { fetchMetaOptions } from "../../../services";
import { fetchTopKUplift, UpliftUnavailableError, type TopKUpliftReport } from "../../../services/topk-uplift";

const ppp = (value: number) => value.toFixed(3);

export default function TopKUpliftPanel() {
  const [seasons, setSeasons] = useState<string[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  const [season, setSeason] = useState("");
  const [our, setOur] = useState("");
  const [opp, setOpp] = useState("");
  const [k, setK] = useState(5);
  const [report, setReport] = useState<TopKUpliftReport | null>(null);
  const [missing, setMissing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  // SCRUM-843: the uplift is computed only for the inputs the analyst ran.
  const [ran, setRan] = useState<{ season: string; our: string; opp: string; k: number } | null>(null);
  const stale = ran !== null && (ran.season !== season || ran.our !== our || ran.opp !== opp || ran.k !== k);

  useEffect(() => {
    fetchMetaOptions().then(meta => {
      // SCRUM-968: an empty fallback means the request failed, so say so instead of leaving blank lists.
      if (meta._fallback || meta.seasons.length === 0) {
        setOptionsError("Seasons and teams didn't load. The server may still be busy with the holdout run.");
        return;
      }
      // SCRUM-841: the analyst chooses the season and teams; none is pre-selected.
      setSeasons(meta.seasons);
      setTeams(meta.teams);
    }).catch(() => setOptionsError("Unable to load seasons and teams."));
  }, []);

  useEffect(() => {
    if (!ran) return;
    const { season, our, opp, k } = ran;
    if (our === opp) { setReport(null); setMissing("Pick two different teams."); return; }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setMissing(null);
    fetchTopKUplift({ season, our, opp, k }, controller.signal).then(result => {
      if (!controller.signal.aborted) setReport(result);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return;
      setReport(null);
      if (reason instanceof UpliftUnavailableError) setMissing(reason.message);
      else setError(reason instanceof Error ? reason.message : "Unable to load uplift.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [ran]);

  const select = (label: string, value: string, options: string[], onChange: (v: string) => void, prompt?: string) => (
    <label style={{ display: "grid", gap: 4 }}>
      <span className="label">{label}</span>
      <select className="input" value={value} onChange={event => onChange(event.target.value)}>
        {prompt ? <option value="" disabled>{prompt}</option> : null}
        {options.map(option => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );

  return <section aria-labelledby="uplift-heading">
    <h2 id="uplift-heading">Top-K uplift</h2>
    <p className="muted">How much better the recommended Top-K play types score than the team&apos;s own season average,
      in points per possession (PPP). Positive means the shortlist beats the team&apos;s usual offense.</p>
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "end" }}>
      {select("Season", season, seasons, setSeason, "Choose a season")}
      {select("Our team", our, teams, setOur, "Choose our team")}
      {select("Opponent", opp, teams, setOpp, "Choose an opponent")}
      {select("Top-K (model setting)", String(k), ["3", "5", "7", "10"], value => setK(Number(value)))}
      <button className="btn primary" type="button" disabled={!season || !our || !opp || our === opp || loading}
        onClick={() => setRan({ season, our, opp, k })}>Compute uplift</button>
    </div>
    {optionsError ? <p role="alert">{optionsError}</p> : null}
    {stale ? <p className="muted" role="status">Inputs changed. Click Compute uplift to update these results.</p> : null}
    {error ? <p role="alert">{error}</p>
      : missing ? <p role="status">No uplift result for this matchup: {missing}</p>
      : !ran ? <p role="status">Choose a season and two teams, then click Compute uplift.</p>
      : loading || !report ? <p role="status">Computing uplift…</p> : <>
        <div className="grid" style={{ marginTop: 12 }}>
          <div className="kpi"><div className="label">Uplift</div>
            <strong>{report.uplift_ppp >= 0 ? "+" : ""}{ppp(report.uplift_ppp)} PPP</strong>
            {report.uplift_percent !== null ? <div className="muted">{report.uplift_percent.toFixed(1)}% vs season average</div> : null}</div>
          <div className="kpi"><div className="label">Top-{report.k_returned} PPP</div><strong>{ppp(report.topk.modeled_ppp)}</strong>
            <div className="muted">Possession-weighted average of the shortlist.</div></div>
          <div className="kpi"><div className="label">Team season PPP</div><strong>{ppp(report.team_season.ppp)}</strong>
            <div className="muted">All of {report.our_team}&apos;s offensive possessions in {report.season}.</div></div>
        </div>
        <div style={{ overflowX: "auto" }}><table className="table">
          <caption>Recommended play types for {report.our_team} vs {report.opp_team}, {report.season}.</caption>
          <thead><tr><th scope="col">Rank</th><th scope="col">Play type</th><th scope="col">Modeled PPP</th><th scope="col">Possessions</th><th scope="col">Uplift (PPP)</th></tr></thead>
          <tbody>{report.rankings.map(row => <tr key={row.rank}>
            <td>{row.rank}</td><td>{row.play_type}</td><td>{ppp(row.modeled_ppp)}</td>
            <td>{row.historical_offense_possessions.toLocaleString()}</td><td>{ppp(row.uplift_vs_team_season_ppp)}</td>
          </tr>)}</tbody>
        </table></div>
        <p className="muted">Scope: {report.evaluation_scope}.</p>
      </>}
  </section>;
}
