"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchMetaOptions, type MetaOptions } from "../../../services/recommendation";
import { fetchPlaytypeDashboard } from "../../../services/playtype-dashboard";
import { DashboardRequest, emptyDashboard, selectDashboard, withRequestDeadline, type DashboardState } from "../../../infrastructure/playtype-dashboard";
import { playtypeName, type DashboardFilters, type PlaytypeDashboard } from "../../../domain/playtype-dashboard";
import { UsageEfficiencyChart, OpponentDefenseChart, signed } from "./PlaytypeCharts";
import LeagueComparisonPanel from "./LeagueComparisonPanel";
import SeasonTrendPanel from "./SeasonTrendPanel";
import TeamComparisonPanel from "./TeamComparisonPanel";
import PossessionOutcomesPanel from "./PossessionOutcomesPanel";
import styles from "./matchup-dashboard.module.css";

type View = "usage" | "defense";
const integer = (n: number) => n.toLocaleString("en-US");

function Sample({ ok, minimum }: { ok: boolean; minimum: number }) {
  return ok ? <span className={styles.sample}>Meets {integer(minimum)} cutoff</span>
    : <span className={styles.smallSample}>Small sample</span>;
}

export function DashboardResults({ data, view }: { data: PlaytypeDashboard; view: View }) {
  const { filters, usage_efficiency: usage, opponent_defense: defense } = data;
  const weakSpots = defense.rows.filter(r => r.weak_spot_rank !== null);
  return <>
    {view === "usage" ? <section className={styles.card} aria-labelledby="usage-title">
      <div className={styles.sectionHeading}>
        <div><h2 id="usage-title">How {filters.our} uses each play type</h2><p>Compare usage and points per possession with the league for the same play type.</p></div>
        <span className={styles.pill}>{integer(usage.total_possessions)} recorded possessions</span>
      </div>
      {usage.rows.length === 0 ? <p role="status">No offensive play-type data is available for this selection.</p> : <>
        <div className={styles.chart}><UsageEfficiencyChart rows={usage.rows} /></div>
        <p className={styles.note}>Numbers match the table. Hollow circles mark small samples. Higher PPP with lower usage is a pattern to review, not proof that using the play more will improve results.</p>
        <div className={styles.tableWrap}><table>
          <caption>{filters.our} usage and efficiency · {filters.season} regular season</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Possessions</th><th scope="col">Usage</th><th scope="col">League usage</th><th scope="col">PPP</th><th scope="col">League PPP</th><th scope="col">Sample</th></tr></thead>
          <tbody>{usage.rows.map((r, i) => <tr key={r.play_type}>
            <th scope="row"><span className={styles.number}>{i + 1}</span>{playtypeName(r.play_type)}</th>
            <td>{integer(r.possessions)}</td><td>{r.usage_pct.toFixed(1)}%</td><td>{r.league_usage_pct.toFixed(1)}%</td>
            <td>{r.ppp.toFixed(3)}</td><td>{r.league_ppp.toFixed(3)}</td><td><Sample ok={r.sample_ok} minimum={filters.min_poss} /></td>
          </tr>)}</tbody>
        </table></div>
      </>}
    </section> : <section className={styles.card} aria-labelledby="defense-title">
      <div className={styles.sectionHeading}>
        <div><h2 id="defense-title">Where {filters.opp} allows more points</h2><p>Ranked by allowed PPP above the league average for each play type.</p></div>
        <span className={styles.pill}>{data.coverage.defense_types} tracked play types</span>
      </div>
      {defense.rows.length === 0 ? <p role="status">No defensive play-type data is available for this selection.</p> : <>
        {weakSpots.length ? <ol className={styles.weakSpots} aria-label="Supported defensive weak spots">
          {weakSpots.map(r => <li key={r.play_type}><span className={styles.kicker}>Area to review {r.weak_spot_rank}</span><strong>{playtypeName(r.play_type)}</strong>
            <span><b>{signed(r.allowed_ppp_delta)} PPP</b> above league</span><small>{integer(r.possessions)} possessions</small></li>)}
        </ol> : <p className={styles.notice} role="status">No above-league defensive weak spot meets this possession cutoff. The full profile is shown below.</p>}
        <div className={styles.chart}><OpponentDefenseChart rows={defense.rows} /></div>
        <p className={styles.note}>Dashed bars mark small samples. Only above-league values that meet the cutoff can be highlighted.</p>
        <div className={styles.tableWrap}><table>
          <caption>{filters.opp} defensive profile · {filters.season} regular season</caption>
          <thead><tr><th scope="col">Play type</th><th scope="col">Possessions</th><th scope="col">Points allowed</th><th scope="col">Allowed PPP</th><th scope="col">League PPP</th><th scope="col">Difference</th><th scope="col">Sample</th></tr></thead>
          <tbody>{defense.rows.map(r => <tr key={r.play_type}><th scope="row">{playtypeName(r.play_type)}</th><td>{integer(r.possessions)}</td>
            <td>{integer(r.points_allowed)}</td><td>{r.allowed_ppp.toFixed(3)}</td><td>{r.league_allowed_ppp.toFixed(3)}</td>
            <td>{signed(r.allowed_ppp_delta)}</td><td><Sample ok={r.sample_ok} minimum={filters.min_poss} /></td></tr>)}</tbody>
        </table></div>
      </>}
      {data.coverage.unavailable_defense_types.length > 0 && <p className={styles.notice}><strong>Defensive data unavailable:</strong> {data.coverage.unavailable_defense_types.map(playtypeName).join(", ")}. These categories are missing from the source, not zero.</p>}
    </section>}
    <details className={styles.details}><summary>How these comparisons work</summary>
      <p>PPP is total points divided by possessions. Usage is each play type’s share of the team’s recorded offensive play-type possessions. League values use the same season, side and play type, weighted by possessions.</p>
      <p>The chart subtracts the league value from the team value. Usage differences are percentage points; efficiency differences are points per possession. The possession cutoff flags small samples without changing the totals or removing rows.</p>
      <p>These are regular-season profiles, not results from games between these two teams or a forecast of their next matchup. The cutoff is a display rule, not a confidence interval.</p>
      <p>Aggregation: <code>{data.aggregation_version}</code></p>
      {Object.entries(data.dataset_versions).map(([source, version]) => <p key={source}>Source version ({source}): <code>{version}</code></p>)}
    </details>
  </>;
}

export function DashboardResultState({ state, view, retry }: { state: DashboardState; view: View; retry: () => void }) {
  if (state.loading) return <div className={styles.card} role="status">Loading the selected teams’ regular-season profiles…</div>;
  if (state.error) return <div className={styles.card} role="alert"><h2>Comparison unavailable</h2><p>{state.error}</p><button className={styles.button} onClick={retry}>Try again</button></div>;
  return state.data ? <DashboardResults data={state.data} view={view} /> : <div className={styles.card} role="status">No comparison data is available.</div>;
}

export default function MatchupDashboardClient() {
  const [meta, setMeta] = useState<MetaOptions | null>(null), [metaError, setMetaError] = useState("");
  const [metaAttempt, setMetaAttempt] = useState(0), [attempt, setAttempt] = useState(0);
  const [season, setSeason] = useState(""), [our, setOur] = useState(""), [opp, setOpp] = useState("");
  const [minimum, setMinimum] = useState(20), [view, setView] = useState<View | "league" | "trend" | "profiles" | "outcomes">("usage");
  const leagueView = view === "league", trendView = view === "trend", profilesView = view === "profiles";
  const outcomesView = view === "outcomes";
  const [playType, setPlayType] = useState("");
  const [leagueSide, setLeagueSide] = useState<"offense" | "defense">("offense");
  const [state, setState] = useState(emptyDashboard);
  const request = useRef<DashboardRequest | null>(null);
  if (!request.current) request.current = new DashboardRequest();
  const valid = Boolean(meta && season && our && opp && our !== opp);

  // SCRUM-843: results appear only after "Show results"; later input changes mark them out of date.
  type Shown = { view: typeof view; season: string; our: string; opp: string; minimum: number; playType: string; leagueSide: "offense" | "defense" };
  const choice: Shown = { view, season, our, opp, minimum, playType, leagueSide };
  const ready = trendView ? Boolean(meta && our && playType) : leagueView || outcomesView ? Boolean(meta && season && our) : valid;
  const [shown, setShown] = useState<Shown | null>(null);
  const stale = shown !== null && JSON.stringify(shown) !== JSON.stringify(choice);
  const shownFilters = useMemo<DashboardFilters | null>(
    () => (shown ? { season: shown.season, our: shown.our, opp: shown.opp, min_poss: shown.minimum } : null),
    [shown],
  );
  const shownTable = Boolean(shown && !["league", "trend", "profiles", "outcomes"].includes(shown.view));

  useEffect(() => {
    let active = true;
    setMetaError("");
    void withRequestDeadline(fetchMetaOptions(), "Team choices took too long to load.").then(result => {
      if (!active) return;
      if (result._fallback || !result.seasons?.length || !result.teams?.length) throw new Error("Team and season choices are unavailable. Please try again.");
      // SCRUM-841: nothing is pre-selected; the coach chooses every input.
      setMeta(result);
    }).catch(() => { if (active) setMetaError("Team and season choices are unavailable. Please try again."); });
    return () => { active = false; };
  }, [metaAttempt]);

  useEffect(() => {
    const current = request.current!;
    if (shownTable && shownFilters) void current.run(shownFilters, fetchPlaytypeDashboard, setState);
    else { current.cancel(); setState(emptyDashboard()); }
    return () => current.cancel();
  }, [shownFilters, shownTable, attempt]);

  return <main className={styles.page}>
    <header className={styles.header}><div><span className={styles.kicker}>Coach workspace · Regular season</span>
      <h1>Matchup dashboard</h1><p>Explore playing style, league comparisons, season trends and defensive weak spots.</p></div>
      <Link className={styles.link} href="/matchup">View baseline rankings →</Link>
    </header>
    {metaError ? <div className={styles.card} role="alert"><p>{metaError}</p><button className={styles.button} onClick={() => setMetaAttempt(x => x + 1)}>Retry team list</button></div>
      : !meta ? <div className={styles.card} role="status">Loading teams and seasons…</div> : <>
        <div className={styles.filters}>
          {!trendView && <label>Season<select value={season} onChange={e => setSeason(e.target.value)}><option value="" disabled>Choose a season</option>{meta.seasons.map(s => <option key={s}>{s}</option>)}</select></label>}
          <label>{leagueView || trendView || outcomesView ? "Team" : "Our team"}<select value={our} onChange={e => setOur(e.target.value)}><option value="" disabled>Choose a team</option>{meta.teams.map(t => <option key={t} value={t}>{t}{meta.teamNames?.[t] ? ` · ${meta.teamNames[t]}` : ""}</option>)}</select></label>
          {trendView ? <label>Play type<select value={playType} onChange={e => setPlayType(e.target.value)}>
            <option value="" disabled>Choose a play type</option>{meta.playTypes?.map(p => <option key={p} value={p}>{playtypeName(p)}</option>)}</select></label>
            : leagueView ? <label>Side<select value={leagueSide} onChange={e => setLeagueSide(e.target.value as "offense" | "defense")}>
            <option value="offense">Offense</option><option value="defense">Defense</option></select></label>
            : !outcomesView ? <label>Opponent<select value={opp} onChange={e => setOpp(e.target.value)}><option value="" disabled>Choose an opponent</option>{meta.teams.map(t => <option key={t} value={t}>{t}{meta.teamNames?.[t] ? ` · ${meta.teamNames[t]}` : ""}</option>)}</select></label> : null}
          <label>{leagueView || trendView || profilesView || outcomesView ? "Sample cutoff" : "Possession cutoff"} (display setting)<select value={minimum} onChange={e => setMinimum(Number(e.target.value))}>{[1, 20, 50, 100, 250, 500, 1000, 5000].map(n => <option key={n} value={n}>{integer(n)}</option>)}</select></label>
        </div>
        <div className={styles.tabs} role="group" aria-label="Choose comparison">
          <button aria-pressed={view === "usage"} onClick={() => setView("usage")}>Usage and efficiency</button>
          <button aria-pressed={view === "defense"} onClick={() => setView("defense")}>Opponent weak spots</button>
          <button aria-pressed={leagueView} onClick={() => setView("league")}>League comparison</button>
          <button aria-pressed={trendView} onClick={() => setView("trend")}>Season trends</button>
          <button aria-pressed={profilesView} onClick={() => setView("profiles")}>Team profiles</button>
          <button aria-pressed={outcomesView} onClick={() => setView("outcomes")}>Possession outcomes</button>
        </div>
        <button className={styles.button} type="button" disabled={!ready} onClick={() => setShown(choice)}>Show results</button>
        {!shown ? <div className={styles.card} role="status" data-testid="dashboard-prompt">
            {trendView ? "Choose a team and a play type" : leagueView || outcomesView ? "Choose a season and a team" : "Choose a season and two different teams"}, then click Show results.
          </div>
          : <>
          {stale ? <p className={styles.notice} role="status">Inputs changed. Click Show results to update these results.</p> : null}
          <p className={styles.scope}>{shown.view === "trend" ? `All available regular seasons · ${shown.our} offense` : `${shown.season} regular season · ${shown.view === "league" ? `${shown.our} ${shown.leagueSide} vs league` : shown.view === "outcomes" ? `${shown.our} offense` : `${shown.our} vs ${shown.opp}`}`} · Historical profiles</p>
          {shown.view === "trend" ? <SeasonTrendPanel filters={{ team: shown.our, play_type: shown.playType }} minimum={shown.minimum} />
          : shown.view === "league" ? <LeagueComparisonPanel filters={{ season: shown.season, team: shown.our, side: shown.leagueSide }} minimum={shown.minimum} />
          : shown.view === "outcomes" ? <PossessionOutcomesPanel filters={{ season: shown.season, team: shown.our }} minimum={shown.minimum} />
          : shown.view === "profiles" ? <>
            <button className={styles.button} type="button" onClick={() => { setOur(shown.opp); setOpp(shown.our); setShown({ ...shown, view, our: shown.opp, opp: shown.our }); }}>Swap teams</button>
            <TeamComparisonPanel filters={{ season: shown.season, team: shown.our, opponent: shown.opp }} minimum={shown.minimum} />
          </>
          : <DashboardResultState state={selectDashboard(state, shownFilters!)} view={shown.view as View} retry={() => setAttempt(x => x + 1)} />}
          </>}
      </>}
  </main>;
}
