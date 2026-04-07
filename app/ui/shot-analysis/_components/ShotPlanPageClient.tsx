"use client";

// Shot Plan (Baseline)
// This page is Dataset2-focused (NBA play-by-play shot data).
// It does two things:
//   1) Calls the backend baseline shot-plan ranker to recommend *what* to shoot.
//   2) Calls the backend heatmap renderer to visualize *where* those shots happen.

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  fetchMetaOptions,
  fetchPbpMetaOptions,
  fetchShotHeatmap,
  fetchShotPlanRank,
  getShotPlanPdfUrl,
  authenticatedDownload,
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
  // Small numeric formatter used throughout tables/KPIs
  const x = Number(n);
  if (Number.isNaN(x)) return "-";
  return x.toFixed(digits);
}

function fmtInt(n: any) {
  // Integer display for attempts/sample sizes
  const x = Number(n);
  if (Number.isNaN(x)) return "-";
  return String(Math.round(x));
}

function pickDefaultSeason(seasons: string[]) {
  // Backend typically returns seasons sorted; safest “latest” is last.
  if (!seasons?.length) return "2021-22";
  return seasons[seasons.length - 1];
}

function pickDefaultTeam(teams: string[], prefer: string) {
  // Prefer TOR/BOS, but fall back safely if not present (prevents empty dropdown edge-cases)
  if (!teams?.length) return prefer;
  return teams.includes(prefer) ? prefer : teams[0];
}

export default function ShotPlanPage() {
  // Meta is merged from:
  // - Dataset1 /meta/options (teamNames mapping: "TOR" -> "Toronto Raptors")
  // - Dataset2 /pbp/meta/options (seasons, teams, shotTypes, zones)
  const [meta, setMeta] = useState<any>(null);

  // Core matchup context
  const [season, setSeason] = useState<string>("2021-22");
  const [our, setOur] = useState<string>("TOR");
  const [opp, setOpp] = useState<string>("BOS");

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

  // Typical request state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Backend responses:
  // - rank: structured baseline recommendations (top_shot_types, top_zones, optional best_shooter)
  // - heatmap: { image_base64, caption?, n_shots?, ... }
  const [rank, setRank] = useState<any>(null);
  const [heatmap, setHeatmap] = useState<any>(null);

  // Load meta options once:
  // - Keep Dataset1 teamNames (nice labels)
  // - Use Dataset2 seasons/teams (so dropdowns match what exists in the shots parquet)
  // - Also pull Dataset2 shotTypes/zones for nicer heatmap filters
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        // Run both meta calls in parallel so the page loads faster
        const [m1, mPbp] = await Promise.all([fetchMetaOptions(), fetchPbpMetaOptions()]);

        // Merge into one meta object:
        // - m1 contains teamNames + any Dataset1 extras
        // - mPbp contains Dataset2-specific dropdown values
        const merged = {
          ...m1,
          seasons: mPbp.seasons,
          teams: mPbp.teams,
          shotTypes: mPbp.shotTypes ?? [],
          zones: mPbp.zones ?? [],
        };

        // If the user navigated away before this resolved, avoid setting state
        if (cancelled) return;
        setMeta(merged);

        // Set safe defaults if state is empty
        // (prevents blank select values if the initial state was "")
        setSeason((prev) => prev || pickDefaultSeason(merged.seasons));
        setOur((prev) => prev || pickDefaultTeam(merged.teams, "TOR"));
        setOpp((prev) => prev || pickDefaultTeam(merged.teams, "BOS"));
      } catch (e: any) {
        // If meta fails, we still want to show the page with an error callout
        if (!cancelled) setError(e?.message ?? "Failed to load meta/options.");
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  // Derived weight for defense side (always complements offense weight)
  const wDef = useMemo(() => 1 - wOff, [wOff]);

  // Helper for showing “TOR (Toronto Raptors)” style labels
  const teamLabel = (abbr: string) => {
    const name = meta?.teamNames?.[abbr];
    return name ? `${abbr} (${name})` : abbr;
  };

  // “Export PDF” uses the existing root endpoint (/export/shotplan.pdf).
  // Notes:
  // - This is not the same as the heatmap endpoint.
  // - It does NOT take maxShots (PDF is explanation + tables, not an image downsampling tool).
  // - We optionally pass shotType/zone so the PDF can match what the user is looking at.
  const pdfUrl = useMemo(() => {
    if (!season || !our || !opp) return "";
    return getShotPlanPdfUrl({
      season,
      our,
      opp,
      k,
      wOff,
      shotType: shotType || undefined,
      zone: zone || undefined,
    });
  }, [season, our, opp, k, wOff, shotType, zone]);

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
    // Basic validation before we even hit the API
    // (keeps the backend clean and gives faster feedback in the UI)
    if (!season || !our || !opp) {
      setError("Please select season, our team, and opponent.");
      return;
    }
    if (our === opp) {
      setError("Our team and opponent cannot be the same.");
      return;
    }
    // Backend expects a small Top-K; UI historically sent large values and broke validation
    if (k < 1 || k > 10) {
      setError("Top K (k) must be between 1 and 10 (backend validation).");
      return;
    }
    if (wOff < 0 || wOff > 1) {
      setError("Offense weight (wOff) must be between 0 and 1.");
      return;
    }

    try {
      setLoading(true);
      setError(null);


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

      // Store responses for rendering
      setRank(r);
      setHeatmap(h);
    } catch (e: any) {
      setError(e?.message ?? "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      {/* “Vibrant” hero-style header using a subtle gradient panel */}
      <header
        className="card"
        style={{
          padding: 20,
          borderRadius: 16,
          background:
            "linear-gradient(135deg, rgba(59,130,246,0.18), rgba(99,102,241,0.10) 45%, rgba(15,22,44,0.95))",
          border: "1px solid rgba(255,255,255,0.10)",
        }}
      >
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <h1 style={{ margin: 0 }}>Shot Plan</h1>
          <span className="badge blue">Dataset2 • PBP Shots</span>
          <span className="badge">Baseline</span>
          <span className="badge">Offense vs Defense</span>
        </div>

        <p className="muted" style={{ marginTop: 8 }}>
          Rank the best <b>shot types</b> and <b>zones</b> for <b>{teamLabel(our)}</b> vs{" "}
          <b>{teamLabel(opp)}</b>, then render a real heatmap for the selected filters.
        </p>

        <p className="muted" style={{ margin: 0 }}>
          Want raw shots? Try <Link href="/shot-explorer">Shots Explorer</Link>.
        </p>
      </header>

      {/* Inputs */}
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>Inputs</h2>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            {/* Show both weights so users understand the blend */}
            <span className="badge">wOff={fmt(wOff, 2)}</span>
            <span className="badge">wDef={fmt(wDef, 2)}</span>

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
        <div className="grid" style={{ marginTop: 12, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          {/* Season */}
          <div>
            <div className="label">Season</div>
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              {(meta?.seasons ?? []).map((s: string) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          {/* Our */}
          <div>
            <div className="label">Our team</div>
            <select className="input" value={our} onChange={(e) => setOur(e.target.value)}>
              {(meta?.teams ?? []).map((t: string) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </div>

          {/* Opp */}
          <div>
            <div className="label">Opponent</div>
            <select className="input" value={opp} onChange={(e) => setOpp(e.target.value)}>
              {(meta?.teams ?? []).map((t: string) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Scoring knobs + heatmap speed knob */}
        <div className="grid" style={{ marginTop: 12, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          {/* Top K */}
          <div>
            <div className="label">Top K results (k)</div>
            <input
              className="input"
              type="number"
              min={1}
              max={10}
              step={1}
              value={k}
              onChange={(e) => setK(Number(e.target.value || "1"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Backend validation: <b>1–10</b>.
            </div>
          </div>

          {/* wOff */}
          <div>
            <div className="label">Offense weight (wOff)</div>
            <input
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
            <div className="label">Heatmap max shots</div>
            <input
              className="input"
              type="number"
              min={1000}
              max={100000}
              step={1000}
              value={maxShots}
              onChange={(e) => setMaxShots(Number(e.target.value || "0"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Downsamples for speed (Dataset2 endpoint supports this).
            </div>
          </div>
        </div>

        {/* Heatmap filters (optional) */}
        <div className="grid" style={{ marginTop: 12, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
          <div>
            <div className="label">Heatmap shot type (optional)</div>
            <select className="input" value={shotType} onChange={(e) => setShotType(e.target.value)}>
              <option value="">(All shot types)</option>
              {(meta?.shotTypes ?? []).map((s: string) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="label">Heatmap zone (optional)</div>
            <select className="input" value={zone} onChange={(e) => setZone(e.target.value)}>
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
          <button className="btn primary" onClick={run} disabled={loading}>
            {loading ? "Running…" : "Run"}
          </button>

          {/* PDF export is a GET URL; opens in new tab */}
          {pdfUrl ? (
            <button
              className="btn"
              type="button"
              onClick={() => authenticatedDownload(pdfUrl).catch(e => alert(e))}
              style={{
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.12)",
              }}
            >
              Export PDF
            </button>
          ) : null}

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

        {/* Inline error callout */}
        {error ? (
          <div className="card" style={{ marginTop: 12, borderColor: "rgba(239,68,68,0.35)" }}>
            <div className="muted" style={{ color: "rgba(239,68,68,0.95)" }}>
              {error}
            </div>
          </div>
        ) : null}
      </section>

      {/* Results */}
      <section className="card">
        <h2>Shot Plan Results</h2>

        {/* Until rank is fetched, show a gentle instruction */}
        {!rank ? (
          <p className="muted">Run the model to see recommended shot plans.</p>
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
              className="grid"
              style={{ marginTop: 12, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}
            >
              <div className="kpi">
                <div className="muted">Best shot type</div>
                <div className="big">{bestShotType?.SHOT_TYPE ?? "-"}</div>
                <div className="muted">
                  EPA_PRED {fmt(bestShotType?.EPA_PRED)} • attempts_OFF{" "}
                  {fmtInt(bestShotType?.attempts_OFF)}
                </div>
              </div>

              <div className="kpi">
                <div className="muted">Best zone</div>
                <div className="big">{bestZone?.ZONE ?? "-"}</div>
                <div className="muted">
                  EPA_PRED {fmt(bestZone?.EPA_PRED)} • attempts_OFF{" "}
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
                    <th>SHOT_TYPE</th>
                    <th>EPA_PRED</th>
                    <th>EPA_OFF_SHRUNK</th>
                    <th>EPA_DEF_SHRUNK</th>
                    <th>ATT_OFF</th>
                    <th>ATT_DEF</th>
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
                    <th>ZONE</th>
                    <th>EPA_PRED</th>
                    <th>EPA_OFF_SHRUNK</th>
                    <th>EPA_DEF_SHRUNK</th>
                    <th>ATT_OFF</th>
                    <th>ATT_DEF</th>
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
          <p className="muted">Run the model to render the shot heatmap.</p>
        ) : (
          <>
            {/* Caption is optional because backend may return only image_base64 */}
            <p className="muted">
              {heatmap.caption
                ? heatmap.caption
                : `Heatmap • ${teamLabel(our)} vs ${teamLabel(opp)} • ${season}`}
              {shotType ? ` • shot_type=${shotType}` : ""}
              {zone ? ` • zone=${zone}` : ""}
            </p>

            <div className="card" style={{ padding: 14 }}>
              {/* image_base64 is returned by backend; we render it as a data URL */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${heatmap.image_base64}`}
                alt="Shot heatmap"
                style={{ width: "100%", maxWidth: 980, height: "auto", borderRadius: 12 }}
              />
              <div className="muted" style={{ marginTop: 10 }}>
                Tip: try narrowing by a <b>zone</b> (e.g. corner 3) to see opponent-specific
                weaknesses.
              </div>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
