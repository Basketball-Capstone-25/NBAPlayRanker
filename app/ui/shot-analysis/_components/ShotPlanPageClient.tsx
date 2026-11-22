"use client";

// Shot Plan (Baseline)
// This page is Dataset2-focused (NBA play-by-play shot data).
// It does two things:
//   1) Calls the backend baseline shot-plan ranker to recommend *what* to shoot.
//   2) Calls the backend heatmap renderer to visualize *where* those shots happen.

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import styles from "../../data-explorer/AnalystWorkspace.module.css";
import { useShotResult } from "./useShotResult";
import { loadShotOptions } from "./shotOptions";
import { ShotHeatmapCounts } from "./ShotHeatmapCounts";
import {
  fetchShotHeatmap,
  fetchShotPlanRank,
  getShotPlanPdfUrl,
  authenticatedDownload,
  type ShotPlanRankResponse,
  type ShotHeatmapResponse,
} from "../../../services/shotAnalysis";

type ShotRow = Record<string, any> & {
  // Depending on the endpoint “level”, the backend may return:
  // - top_shot_types rows with SHOT_TYPE populated
  // - top_zones rows with ZONE populated
  // We keep both optional so the same type works for both tables.
  SHOT_TYPE?: string | null;
  ZONE?: string | null;

  // Core numeric fields the backend emits (all optional because backend may omit fields in future)
  // EPA_PRED: blended score (offense + opponent defense) used for ranking
  // EPA_OFF_SHRUNK: offense-only expected points estimate with shrinkage / reliability
  // EPA_DEF_SHRUNK: opponent defense estimate with shrinkage / reliability
  // attempts_OFF / attempts_DEF: sample sizes used to stabilize estimates
  EPA_PRED?: number;
  EPA_OFF_SHRUNK?: number;
  EPA_DEF_SHRUNK?: number;
  attempts_OFF?: number;
  attempts_DEF?: number;

  // Reason text returned by backend.
  RATIONALE?: string;
};

function fmt(n: any, digits = 3) {
  if (n == null || n === "") return "Unavailable";
  const x = Number(n);
  if (!Number.isFinite(x)) return "Unavailable";
  return x.toFixed(digits);
}

function fmtInt(n: any) {
  if (n == null || n === "") return "Unavailable";
  const x = Number(n);
  if (!Number.isFinite(x)) return "Unavailable";
  return String(Math.round(x));
}

export default function ShotPlanPage() {
  // Meta is merged from:
  // - Dataset1 /meta/options (teamNames mapping: "TOR" -> "Toronto Raptors")
  // - Dataset2 /pbp/meta/options (seasons, teams, shotTypes, zones)
  const [meta, setMeta] = useState<any>(null);

  // Core matchup context
  const [season, setSeason] = useState<string>("");
  const [our, setOur] = useState<string>("");
  const [opp, setOpp] = useState<string>("");

  // Backend expects k between 1 and 10.
  const [k, setK] = useState<number>(5);

  // Blending between our offense signal and opponent defense signal.
  // wOff = 1.0 => offense-only
  // wOff = 0.0 => opponent-defense-only
  // wDef is derived as (1 - wOff)
  const [wOff, setWOff] = useState<number>(0.7);

  // Optional heatmap filters.
  // These do NOT affect the ranking call (rank is always “global” for the matchup).
  // They only affect the rendered heatmap image.
  const [shotType, setShotType] = useState<string>("");
  const [zone, setZone] = useState<string>("");

  // Dataset2 heatmap endpoint supports downsampling via maxShots (speed vs fidelity).
  const [maxShots, setMaxShots] = useState<number>(35000);

  // Toggle to show reason text for the top pick.
  const [showWhy, setShowWhy] = useState<boolean>(false);

  const [inputError, setInputError] = useState<string | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [metadataAttempt, setMetadataAttempt] = useState(0);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const inputs = { season, our, opp, k, wOff, shotType, zone, maxShots };
  const query = useShotResult<typeof inputs, { rank: ShotPlanRankResponse; heatmap: ShotHeatmapResponse }>(inputs);
  const { loading } = query;
  const error = metadataError ?? inputError ?? query.error;
  const rank = !error ? query.result?.rank : null;
  const heatmap = !error ? query.result?.heatmap : null;

  // Load meta options once:
  // - Keep Dataset1 teamNames (nice labels)
  // - Use Dataset2 seasons/teams (so dropdowns match what exists in the shots parquet)
  // - Also pull Dataset2 shotTypes/zones for nicer heatmap filters
  useEffect(() => {
    let cancelled = false;
    setMetadataError(null);

    async function init() {
      try {
        const merged = await loadShotOptions();

        // If the user navigated away before this resolved, avoid setting state
        if (cancelled) return;
        setMeta(merged);

        // SCRUM-841: nothing is pre-selected.
      } catch (e: any) {
        // If meta fails, we still want to show the page with an error callout
        if (!cancelled) setMetadataError(e?.message ?? "Shot filters could not be loaded.");
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [metadataAttempt]);

  // Derived weight for defense side (always complements offense weight)
  const wDef = useMemo(() => 1 - wOff, [wOff]);

  // Helper for showing “TOR (Toronto Raptors)” style labels
  const teamLabel = (abbr: string) => {
    const name = meta?.teamNames?.[abbr];
    return name ? `${abbr} (${name})` : abbr;
  };

  const pdfUrl = rank && query.inputs ? getShotPlanPdfUrl({ ...query.inputs,
    shotType: query.inputs.shotType || undefined, zone: query.inputs.zone || undefined,
  }) : "";

  async function exportPdf() {
    if (!pdfUrl || exporting) return;
    setExporting(true); setExportError(null);
    try { await authenticatedDownload(pdfUrl); }
    catch (reason) { setExportError(reason instanceof Error ? reason.message : "The PDF could not be downloaded."); }
    finally { setExporting(false); }
  }

  // Convenience: derive “quick picks” so the results feel more like a “gameplan”
  // Backend shape for rank is expected to have:
  // - top_shot_types: array of rows (SHOT_TYPE + metrics)
  // - top_zones: array of rows (ZONE + metrics)
  const topShotTypes: ShotRow[] = rank?.top_shot_types ?? [];
  const topZones: ShotRow[] = rank?.top_zones ?? [];

  // Display-friendly “best” items (first row is highest-ranked)
  const bestShotType = topShotTypes?.[0];
  const bestZone = topZones?.[0];

  async function run() {
    if (!meta) return;
    setInputError(null); setExportError(null);
    if (!season || !our || !opp) {
      setInputError("Please select season, our team, and opponent.");
      return;
    }
    if (our === opp) {
      setInputError("Our team and opponent cannot be the same.");
      return;
    }
    // Backend expects a small Top-K; UI historically sent large values and broke validation
    if (!Number.isInteger(k) || k < 1 || k > 10) {
      setInputError("Choose a whole number from 1 to 10 for Top K.");
      return;
    }
    if (!Number.isFinite(wOff) || wOff < 0 || wOff > 1) {
      setInputError("Offense weight must be between 0 and 1.");
      return;
    }
    if (!Number.isInteger(maxShots) || maxShots < 1000 || maxShots > 100000) {
      setInputError("Heatmap max shots must be a whole number from 1,000 to 100,000.");
      return;
    }
    await query.run(async ({ season, our, opp, k, wOff, shotType, zone, maxShots }) => {
      const [r, h] = await Promise.all([
        fetchShotPlanRank({ season, our, opp, k, wOff }),
        fetchShotHeatmap({
          season,
          team: our, // Dataset2 endpoint uses `team` (utils supports team OR our)
          opp,
          shotType: shotType || undefined,
          zone: zone || undefined,
          maxShots,
        }),
      ]);

      return { rank: r, heatmap: h };
    });
  }

  return (
    <div className={`page ${styles.workspace}`}>
      {/* “Vibrant” hero-style header using a subtle gradient panel */}
      <header className={styles.header}>
        <div className={styles.eyebrow}>Shot analysis</div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <h1 style={{ margin: 0 }}>Shot Plan</h1>

          <span className="badge">Baseline</span>

        </div>

        <p className="muted" style={{ marginTop: 8 }}>
          Compare shot types and zones using season-long offense and defense, including regular-season and playoff games.
        </p>
        <p className={styles.notice}>Scores are estimated points per shot attempt, not points per possession. The heatmap shows only shots taken against the selected opponent; its filters do not change the rankings.</p>

        <p className="muted" style={{ margin: 0 }}>
          Browse individual attempts in <Link href="/shot-explorer">Shots Explorer</Link>.
        </p>
      </header>

      {/* Inputs */}
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>Inputs</h2>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            {/* Show both weights so users understand the blend */}
            <span className="badge">Offense {Math.round(wOff * 100)}%</span>
            <span className="badge">Defense {Math.round(wDef * 100)}%</span>

            {/* Toggle “why” rationale blocks (from backend RATIONALE field) */}
            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={showWhy}
                onChange={(e) => setShowWhy(e.target.checked)}
              />
              <span className="muted">Show “Why” explanations</span>
            </label>
          </div>
        </div>

        {/* Matchup selectors */}
        <div className={styles.columns}>
          {/* Season */}
          <div>
            <label className="label" htmlFor="shot-plan-input-1">Season</label>
            <select id="shot-plan-input-1" className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              <option value="" disabled>Choose a season</option>
              {(meta?.seasons ?? []).map((s: string) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          {/* Our */}
          <div>
            <label className="label" htmlFor="shot-plan-input-2">Our team</label>
            <select id="shot-plan-input-2" className="input" value={our} onChange={(e) => setOur(e.target.value)}>
              <option value="" disabled>Choose our team</option>
              {(meta?.teams ?? []).map((t: string) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </div>

          {/* Opp */}
          <div>
            <label className="label" htmlFor="shot-plan-input-3">Opponent</label>
            <select id="shot-plan-input-3" className="input" value={opp} onChange={(e) => setOpp(e.target.value)}>
              <option value="" disabled>Choose an opponent</option>
              {(meta?.teams ?? []).map((t: string) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Scoring knobs + heatmap speed knob */}
        <div className={styles.columns}>
          {/* Top K */}
          <div>
            <label className="label" htmlFor="shot-plan-input-4">Top K results (k)</label>
            <input id="shot-plan-input-4"
              className="input"
              type="number"
              min={1}
              max={10}
              step={1}
              value={k}
              onChange={(e) => setK(Number(e.target.value || "1"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Show up to <b>10</b> results in each table.
            </div>
          </div>

          {/* wOff */}
          <div>
            <label className="label" htmlFor="shot-plan-input-5">Offense weight (wOff)</label>
            <input id="shot-plan-input-5"
              className="input"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={wOff}
              onChange={(e) => setWOff(Number(e.target.value || "0"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Higher = trust our offense more. Lower = trust opponent defense more.
            </div>
          </div>

          {/* maxShots */}
          <div>
            <label className="label" htmlFor="shot-plan-input-6">Heatmap max shots</label>
            <input id="shot-plan-input-6"
              className="input"
              type="number"
              min={1000}
              max={100000}
              step={1000}
              value={maxShots}
              onChange={(e) => setMaxShots(Number(e.target.value || "0"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Samples the heatmap when more attempts match.
            </div>
          </div>
        </div>

        {/* Heatmap filters (optional) */}
        <div className={styles.columns}>
          <div>
            <label className="label" htmlFor="shot-plan-input-7">Heatmap shot type (optional)</label>
            <select id="shot-plan-input-7" className="input" value={shotType} onChange={(e) => setShotType(e.target.value)}>
              <option value="">(All shot types)</option>
              {(meta?.shotTypes ?? []).map((s: string) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label" htmlFor="shot-plan-input-8">Heatmap zone (optional)</label>
            <select id="shot-plan-input-8" className="input" value={zone} onChange={(e) => setZone(e.target.value)}>
              <option value="">(All zones)</option>
              {(meta?.zones ?? []).map((z: string) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
          <button className="btn primary" onClick={run} disabled={loading || !meta}>
            {loading ? "Running…" : "Run"}
          </button>

            <button
              className="btn"
              type="button"
              onClick={exportPdf}
              disabled={!pdfUrl || exporting}
              style={{
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.12)",
              }}
            >
              {exporting ? "Preparing PDF…" : "Export PDF"}
            </button>

          {/* Separate page for deep heatmap exploration */}
          <Link
            className="btn"
            href="/shot-heatmap"
            style={{
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.12)",
            }}
          >
            Open Heatmap Page
          </Link>
        </div>
        {query.changed && <p role="status">Inputs changed. Run again to see and export the updated results.</p>}
        {exportError && <p role="alert">{exportError}</p>}

        {/* Inline error callout */}
        {error ? (
          <div className="card" style={{ marginTop: 12, borderColor: "rgba(239,68,68,0.35)" }}>
            <div role="alert" className="muted" style={{ color: "rgba(239,68,68,0.95)" }}>
              {error}
              {metadataError && <button className="btn btn--secondary" type="button" onClick={() => setMetadataAttempt(value => value + 1)}>Retry filters</button>}
            </div>
          </div>
        ) : null}
      </section>

      {/* Results */}
      <section className="card">
        <h2>Shot Plan Results</h2>

        {/* Until rank is fetched, show a gentle instruction */}
        {!rank ? (
          <p className="muted">{loading ? "Loading shot plan…" : "Run to see recommended shot types and zones."}</p>
        ) : (
          <>
            {/* Backend returns canonical keys: season, our_team, opp_team, k, w_off, w_def */}
            <p className="muted">
              Season <b>{rank.season}</b> • <b>{teamLabel(rank.our_team)}</b> vs{" "}
              <b>{teamLabel(rank.opp_team)}</b> • k={rank.k} • wOff={fmt(rank.w_off, 2)} • wDef=
              {fmt(rank.w_def, 2)}
            </p>

            {/* Quick summary cards */}
            <div
              className={styles.columns}
            >
              <div className="kpi">
                <div className="muted">Best shot type</div>
                <div className="big">{bestShotType?.SHOT_TYPE ?? "-"}</div>
                <div className="muted">
                  Estimated points/attempt {fmt(bestShotType?.EPA_PRED)} • Our attempts{" "}
                  {fmtInt(bestShotType?.attempts_OFF)}
                </div>
              </div>

              <div className="kpi">
                <div className="muted">Best zone</div>
                <div className="big">{bestZone?.ZONE ?? "-"}</div>
                <div className="muted">
                  Estimated points/attempt {fmt(bestZone?.EPA_PRED)} • Our attempts{" "}
                  {fmtInt(bestZone?.attempts_OFF)}
                </div>
              </div>

              <div className="kpi">
                <div className="muted">Best shooter (optional)</div>
                <div className="big">{rank?.best_shooter?.PLAYER_NAME ?? "-"}</div>
                <div className="muted">
                  {rank?.best_shooter?.ROLE ? `Role: ${rank.best_shooter.ROLE}` : "—"}
                </div>
              </div>
            </div>

            {/* Top shot types table */}
            <h3 style={{ marginTop: 18 }}>Top Shot Types</h3>
            <div className="card" style={{ padding: 0, overflowX: "auto" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Shot type</th>
                    <th>Estimated points/attempt</th>
                    <th>Our adjusted points/attempt</th>
                    <th>Opponent allowed points/attempt</th>
                    <th>Our attempts</th>
                    <th>Opponent defended attempts</th>
                  </tr>
                </thead>
                <tbody>
                  {topShotTypes.map((r, idx) => (
                    <tr key={idx}>
                      <td>{r.SHOT_TYPE}</td>
                      <td>{fmt(r.EPA_PRED)}</td>
                      <td>{fmt(r.EPA_OFF_SHRUNK)}</td>
                      <td>{fmt(r.EPA_DEF_SHRUNK)}</td>
                      <td>{fmtInt(r.attempts_OFF)}</td>
                      <td>{fmtInt(r.attempts_DEF)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Optional “why” block for the top-ranked shot type */}
            {showWhy && topShotTypes?.length ? (
              <div className="card" style={{ marginTop: 12 }}>
                <div className="label">Why this shot type is recommended</div>
                <div className="muted" style={{ marginTop: 6 }}>
                  {topShotTypes[0]?.RATIONALE ?? "No rationale returned."}
                </div>
              </div>
            ) : null}

            {/* Top zones table */}
            <h3 style={{ marginTop: 18 }}>Top Zones</h3>
            <div className="card" style={{ padding: 0, overflowX: "auto" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Zone</th>
                    <th>Estimated points/attempt</th>
                    <th>Our adjusted points/attempt</th>
                    <th>Opponent allowed points/attempt</th>
                    <th>Our attempts</th>
                    <th>Opponent defended attempts</th>
                  </tr>
                </thead>
                <tbody>
                  {topZones.map((r, idx) => (
                    <tr key={idx}>
                      <td>{r.ZONE}</td>
                      <td>{fmt(r.EPA_PRED)}</td>
                      <td>{fmt(r.EPA_OFF_SHRUNK)}</td>
                      <td>{fmt(r.EPA_DEF_SHRUNK)}</td>
                      <td>{fmtInt(r.attempts_OFF)}</td>
                      <td>{fmtInt(r.attempts_DEF)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Optional “why” block for the top-ranked zone */}
            {showWhy && topZones?.length ? (
              <div className="card" style={{ marginTop: 12 }}>
                <div className="label">Why this zone is recommended</div>
                <div className="muted" style={{ marginTop: 6 }}>
                  {topZones[0]?.RATIONALE ?? "No rationale returned."}
                </div>
              </div>
            ) : null}
          </>
        )}
      </section>

      {/* Heatmap */}
      <section className="card">
        <h2>Heatmap</h2>

        {/* Heatmap is only available after Run() */}
        {!heatmap ? (
          <p className="muted">{loading ? "Loading heatmap…" : "Run to display shot attempts for this matchup."}</p>
        ) : (
          <>
            {/* Caption is optional because backend may return only image_base64 */}
            <p className="muted">
              {heatmap.caption
                ? heatmap.caption
                : `Heatmap • ${teamLabel(query.inputs!.our)} vs ${teamLabel(query.inputs!.opp)} • ${query.inputs!.season}`}
            </p>

            <ShotHeatmapCounts heatmap={heatmap} />

            <div className="card" style={{ padding: 14 }}>
              {/* image_base64 is returned by backend; we render it as a data URL */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${heatmap.image_base64}`}
                alt="Shot heatmap"
                style={{ width: "100%", maxWidth: 980, height: "auto", borderRadius: 12 }}
              />
              <div className="muted" style={{ marginTop: 10 }}>
                The colors show where attempts occurred, including makes and misses. They do not measure shooting percentage or defensive weakness.
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
