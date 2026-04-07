// Baseline matchup page UI.

"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  authenticatedDownload,
  baselineRank,
  fetchBaselineInfo,
  fetchMetaOptions,
  fetchPlaytypeViz,
  getBaselineCsvUrl,
} from "../../../services/recommendation";
import { normalizeWeights } from "../../../domain/weight-calculation";

type MetaOptions = {
  seasons: string[];
  teams: string[];
  teamNames?: Record<string, string>;
  playTypes?: string[];
  sides?: string[];
  hasMlPredictions?: boolean;
  _fallback?: boolean;
};

type BaselineInfo = {
  formula?: string;
  defaults?: { w_off?: number; w_def?: number };
  definitions?: Record<string, string>;
  whyShrinkage?: string;
};

type BaselineRow = {
  playType: string;
  pppPred: number;
  pppOff: number;
  pppDef: number;
  pppGap: number;
  rationale: string;
  raw: Record<string, any>;
};

type VizResponse = {
  caption: string;
  image_base64: string;
};
function fmt(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}
function fmtInt(n: any) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return String(Math.round(x));
}
function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}
function safePct(x: any) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

// Defaults make the page feel “ready” on load
function pickDefaultSeason(seasons: string[]) {
  return seasons.length ? seasons[seasons.length - 1] : "";
}
function pickDefaultOur(teams: string[]) {
  if (teams.includes("TOR")) return "TOR";
  return teams[0] ?? "";
}
function pickDefaultOpp(teams: string[], our: string) {
  const preferred = ["BOS", "LAL", "DEN", "MIA", "GSW"];
  for (const t of preferred) {
    if (teams.includes(t) && t !== our) return t;
  }
  return teams.find((t) => t !== our) ?? "";
}

function Icon({ name }: { name: "play" | "download" | "map" | "pdf" | "info" }) {
  const common = {
    width: 16,
    height: 16,
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
  };
  switch (name) {
    case "play":
      return (
        <svg {...common}>
          <path d="M10 8.5v7l6-3.5-6-3.5Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" stroke="currentColor" strokeWidth="2" />
        </svg>
      );
    case "download":
      return (
        <svg {...common}>
          <path d="M12 3v10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M8 10l4 4 4-4" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M5 21h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    case "map":
      return (
        <svg {...common}>
          <path
            d="M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path d="M9 3v15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M15 6v15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    case "pdf":
      return (
        <svg {...common}>
          <path d="M7 3h7l3 3v15H7V3Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M14 3v4h4" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M9 14h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M9 17h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z" stroke="currentColor" strokeWidth="2" />
          <path d="M12 10v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M12 7h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
  }
}

export default function MatchupPage() {
  // Meta drives dropdowns (seasons/teams)
  const [meta, setMeta] = useState<MetaOptions>({
    seasons: [],
    teams: [],
    teamNames: {},
    playTypes: [],
    sides: ["offense", "defense"],
    hasMlPredictions: false,
  });

  // Extra notes shown in the details section.
  const [baselineInfo, setBaselineInfo] = useState<BaselineInfo | null>(null);

  // Inputs
  const [season, setSeason] = useState("");
  const [our, setOur] = useState("");
  const [opp, setOpp] = useState("");
  const [k, setK] = useState(5);

  // Weights
  const [wOff, setWOff] = useState(0.7);
  const [wDef, setWDef] = useState(0.3);

  // Results
  const [rows, setRows] = useState<BaselineRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Optional: show extra math fields in the details section.
  const [showMath, setShowMath] = useState(false);

  // Viz state
  const [vizPlayType, setVizPlayType] = useState<string>("");
  const [viz, setViz] = useState<VizResponse | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizError, setVizError] = useState<string | null>(null);

  const didAutoRunRef = useRef(false);

  // Init meta + defaults
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setError(null);

        const m = await fetchMetaOptions();
        if (cancelled) return;
        setMeta(m);

        const defaultSeason = pickDefaultSeason(m.seasons ?? []);
        const defaultOur = pickDefaultOur(m.teams ?? []);
        const defaultOpp = pickDefaultOpp(m.teams ?? [], defaultOur);

        setSeason((prev) => prev || defaultSeason);
        setOur((prev) => prev || defaultOur);
        setOpp((prev) => prev || defaultOpp);

        const b = await fetchBaselineInfo();
        if (cancelled) return;
        setBaselineInfo(b);

        if (b?.defaults?.w_off != null) setWOff(Number(b.defaults.w_off));
        if (b?.defaults?.w_def != null) setWDef(Number(b.defaults.w_def));
      } catch (e: any) {
        console.error(e);
        if (!cancelled) setError(e?.message ?? "Failed to load metadata.");
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  const metaReady = useMemo(
    () => (meta?.seasons?.length ?? 0) > 0 && (meta?.teams?.length ?? 0) > 0,
    [meta]
  );

  const norm = useMemo(() => normalizeWeights(wOff, wDef), [wOff, wDef]);

  const canRun = useMemo(() => {
    if (!metaReady) return false;
    if (!season || !our || !opp) return false;
    if (our === opp) return false;
    return true;
  }, [metaReady, season, our, opp]);

  const csvUrl = useMemo(() => {
    if (!season || !our || !opp) return "#";
    return getBaselineCsvUrl({ season, our, opp, k, wOff: norm.wOff, wDef: norm.wDef });
  }, [season, our, opp, k, norm.wOff, norm.wDef]);

  const runContextHref = useMemo(() => {
    const qs = new URLSearchParams({ season, our, opp, k: String(k) });
    return `/context?${qs.toString()}`;
  }, [season, our, opp, k]);

  const matchupLabel = useMemo(() => {
    const ourName = meta.teamNames?.[our];
    const oppName = meta.teamNames?.[opp];
    const left = ourName ? `${our} (${ourName})` : our;
    const right = oppName ? `${opp} (${oppName})` : opp;
    return `${left} vs ${right}`;
  }, [meta.teamNames, our, opp]);

  // Reset viz when matchup changes
  useEffect(() => {
    setViz(null);
    setVizError(null);
    setVizPlayType("");
  }, [season, our, opp, k, wOff, wDef]);

  function getPdfUrlForPlayType(playType: string) {
    if (!season || !our || !opp || !playType) return "#";
    const params = new URLSearchParams({
      season,
      our,
      opp,
      play_type: playType,
      k: String(k),
      w_off: String(norm.wOff),
    });
    return `${API_BASE}/export/playtype-viz.pdf?${params.toString()}`;
  }

  async function runBaseline() {
    if (!season || !our || !opp) {
      setError("Please select a season, our team, and an opponent.");
      return;
    }
    if (our === opp) {
      setError("Our team and opponent must be different.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setRows([]);

      const out = await baselineRank({
        season,
        our,
        opp,
        k,
        wOff: norm.wOff,
        wDef: norm.wDef,
      });

      setRows(Array.isArray(out) ? out : []);
    } catch (e: any) {
      console.error(e);
      setError(e?.message ?? "Failed to generate baseline recommendations.");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  async function runViz(playTypeOverride?: string) {
    const pt = (playTypeOverride ?? vizPlayType)?.trim();
    if (!pt) {
      setVizError("Pick a play type first.");
      return;
    }
    if (!season || !our || !opp) {
      setVizError("Select season/teams first.");
      return;
    }

    try {
      setVizLoading(true);
      setVizError(null);
      setViz(null);

      const out = await fetchPlaytypeViz({
        season,
        our,
        opp,
        playType: pt,
        wOff: norm.wOff,
      });

      setViz(out);
    } catch (e: any) {
      console.error(e);
      setVizError(e?.message ?? "Failed to generate court visualization.");
      setViz(null);
    } finally {
      setVizLoading(false);
    }
  }

  // Auto-run once when defaults are loaded
  useEffect(() => {
    if (!metaReady) return;
    if (!season || !our || !opp) return;
    if (didAutoRunRef.current) return;
    if (loading) return;

    didAutoRunRef.current = true;
    runBaseline();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metaReady, season, our, opp]);

  const bestRow = useMemo(() => (rows.length ? rows[0] : null), [rows]);

  // Default viz play type to #1
  useEffect(() => {
    if (!rows.length) return;
    if (vizPlayType) return;
    if (bestRow?.playType) setVizPlayType(bestRow.playType);
  }, [rows, bestRow, vizPlayType]);

  const vizReady = useMemo(() => Boolean(season && our && opp && vizPlayType), [season, our, opp, vizPlayType]);

  return (
    <section className="pageRoot">
      <style jsx>{`
        .pageRoot {
          max-width: 1100px;
          margin: 0 auto;
          padding: 18px 16px 34px;
        }

        .hero {
          display: grid;
          gap: 6px;
          margin-bottom: 12px;
        }

        .heroTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(15, 23, 42, 0.03);
          color: rgba(15, 23, 42, 0.75);
          font-size: 12px;
          white-space: nowrap;
        }

        h1 {
          margin: 0;
          font-size: 22px;
          letter-spacing: -0.3px;
        }

        .sub {
          margin: 0;
          color: rgba(15, 23, 42, 0.7);
          font-size: 13px;
          line-height: 1.45;
        }

        .card {
          border-radius: 16px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(255, 255, 255, 0.78);
          padding: 14px;
        }

        .grid2 {
          display: grid;
          grid-template-columns: 1fr;
          gap: 12px;
        }
        @media (min-width: 860px) {
          .grid2 {
            grid-template-columns: 1.05fr 0.95fr;
            align-items: start;
          }
        }

        .sectionTitle {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }
        .sectionTitle h2 {
          margin: 0;
          font-size: 14px;
          letter-spacing: -0.2px;
        }

        .inputs {
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
        }
        @media (min-width: 640px) {
          .inputs {
            grid-template-columns: 1fr 1fr;
          }
        }

        label {
          display: grid;
          gap: 6px;
          font-size: 12px;
          color: rgba(15, 23, 42, 0.8);
        }

        .input {
          width: 100%;
          border-radius: 12px;
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: rgba(255, 255, 255, 0.95);
          padding: 10px 10px;
          font-size: 13px;
          color: rgba(15, 23, 42, 0.92);
          outline: none;
        }

        .row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(15, 23, 42, 0.08);
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.14);
          background: rgba(255, 255, 255, 0.9);
          color: rgba(15, 23, 42, 0.92);
          font-weight: 800;
          font-size: 13px;
          text-decoration: none;
          cursor: pointer;
        }

        .btnPrimary {
          border: 1px solid rgba(255, 255, 255, 0.26);
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          color: #fff;
          box-shadow: 0 16px 34px rgba(37, 99, 235, 0.22);
          text-shadow: 0 1px 0 rgba(0, 0, 0, 0.14);
        }

        .btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }

        .muted {
          color: rgba(15, 23, 42, 0.68);
          font-size: 12px;
          margin: 0;
        }

        .error {
          margin-top: 10px;
          font-size: 12px;
          color: rgba(185, 28, 28, 0.92);
        }

        .weightsWrap {
          display: grid;
          gap: 10px;
          margin-top: 12px;
        }

        .weightBox {
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(15, 23, 42, 0.02);
          border-radius: 14px;
          padding: 10px;
          display: grid;
          gap: 8px;
        }

        .weightTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
        }

        .range {
          width: 100%;
        }

        .resultsCard {
          margin-top: 12px;
        }

        .topRec {
          border: 1px solid rgba(15, 23, 42, 0.1);
          border-radius: 14px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.9);
        }

        .topRecTitle {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
        }

        .playName {
          font-weight: 900;
          font-size: 15px;
          letter-spacing: -0.2px;
        }

        .barList {
          display: grid;
          gap: 10px;
          margin-top: 12px;
        }

        .barRow {
          display: grid;
          grid-template-columns: 1fr;
          gap: 8px;
        }

        @media (min-width: 720px) {
          .barRow {
            grid-template-columns: 240px 1fr 80px;
            align-items: center;
            gap: 10px;
          }
        }

        .barLabel {
          display: grid;
          gap: 2px;
        }

        .meter {
          height: 10px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.1);
          overflow: hidden;
        }

        .meter > div {
          height: 100%;
          background: rgba(15, 23, 42, 0.35);
        }

        .mono {
          font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace);
          font-size: 12px;
        }

        details.moreInfo {
          margin-top: 14px;
          border-top: 1px solid rgba(15, 23, 42, 0.08);
          padding-top: 12px;
        }
        details.moreInfo summary {
          cursor: pointer;
          font-weight: 900;
          color: rgba(15, 23, 42, 0.9);
          list-style: none;
        }
        details.moreInfo summary::-webkit-details-marker {
          display: none;
        }

        .tableWrap {
          overflow-x: auto;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(255, 255, 255, 0.86);
          margin-top: 10px;
        }

        table {
          width: max-content;
          min-width: 100%;
          border-collapse: collapse;
        }

        th,
        td {
          padding: 10px 10px;
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          text-align: left;
          font-size: 12px;
          white-space: nowrap;
        }

        th {
          font-weight: 900;
          color: rgba(15, 23, 42, 0.82);
          background: rgba(15, 23, 42, 0.02);
        }

        .imgFrame {
          width: 100%;
          overflow: hidden;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: rgba(15, 23, 42, 0.02);
          margin-top: 10px;
        }
        .imgFrame img {
          width: 100%;
          height: auto;
          display: block;
        }
      `}</style>

      {/* Header */}
      <div className="hero">
        <div className="heroTop">
          <div className="badge">
            <Icon name="info" /> Baseline • Coach view
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href="/data-explorer">
              Data Explorer
            </Link>
            <Link className="btn" href={runContextHref}>
              Next: Context / ML
            </Link>
          </div>
        </div>

        <h1>Matchup / Baseline</h1>
        <p className="sub">
          Pick a matchup and get the Top-K play types. This is the transparent “trust anchor” before Context / ML.
        </p>
      </div>

      {/* Inputs + Run */}
      <div className="card">
        <div className="sectionTitle">
          <h2>Matchup + context inputs</h2>
          <span className="badge">{metaReady ? "Ready" : "Loading…"}</span>
        </div>

        <div className="inputs">
          <label>
            Season
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              {(meta.seasons ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Top-K
            <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[3, 5, 7, 10].map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
          </label>

          <label>
            Our Team
            <select className="input" value={our} onChange={(e) => setOur(e.target.value)}>
              {(meta.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select className="input" value={opp} onChange={(e) => setOpp(e.target.value)}>
              {(meta.teams ?? []).map((t) => (
                <option key={t} value={t} disabled={t === our}>
                  {t}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="weightsWrap">
          <div className="sectionTitle" style={{ marginBottom: 0 }}>
            <h2>Weights (auto-normalized)</h2>
            <span className="badge">
              off={fmt(norm.wOff, 2)} • def={fmt(norm.wDef, 2)}
            </span>
          </div>

          <div className="grid2">
            <div className="weightBox">
              <div className="weightTop">
                <div>
                  <div style={{ fontWeight: 900, fontSize: 12 }}>Offense weight</div>
                  <p className="muted">Controls how much our offense drives the ranking.</p>
                </div>
                <input
                  className="input"
                  style={{ width: 90 }}
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={wOff}
                  onChange={(e) => setWOff(Number(e.target.value))}
                />
              </div>
              <input
                className="range"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={Number.isFinite(wOff) ? wOff : 0}
                onChange={(e) => setWOff(Number(e.target.value))}
              />
            </div>

            <div className="weightBox">
              <div className="weightTop">
                <div>
                  <div style={{ fontWeight: 900, fontSize: 12 }}>Defense weight</div>
                  <p className="muted">Controls how much opponent defense drives the ranking.</p>
                </div>
                <input
                  className="input"
                  style={{ width: 90 }}
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={wDef}
                  onChange={(e) => setWDef(Number(e.target.value))}
                />
              </div>
              <input
                className="range"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={Number.isFinite(wDef) ? wDef : 0}
                onChange={(e) => setWDef(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row">
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button className="btn btnPrimary" type="button" onClick={runBaseline} disabled={!canRun || loading}>
                <Icon name="play" /> {loading ? "Running…" : "Run Baseline"}
              </button>

              <button className="btn" type="button" disabled={!canRun} onClick={() => authenticatedDownload(csvUrl).catch(e => setError(String(e)))}>
                <Icon name="download" /> Export CSV
              </button>
            </div>

            <div className="badge">{season && our && opp ? matchupLabel : "Select teams"}</div>
          </div>

          {error ? <div className="error">{error}</div> : null}
        </div>
      </div>

      {/* Results */}
      <div className="card resultsCard">
        <div className="sectionTitle">
          <h2>Top recommendations</h2>
          <span className="badge">{rows.length ? `${our} vs ${opp} • ${season}` : "Run to see results"}</span>
        </div>

        {!loading && rows.length === 0 ? (
          <p className="muted">Pick teams + season, then click Run Baseline.</p>
        ) : loading ? (
          <p className="muted">Computing shrunk PPPs and ranking Top-K…</p>
        ) : (
          <>
            {/* Top 1 */}
            {bestRow ? (
              <div className="topRec">
                <div className="topRecTitle">
                  <div className="playName">
                    #{1} {bestRow.playType}
                  </div>
                  <div className="badge">
                    Pred PPP <span className="mono">{fmt(bestRow.pppPred, 3)}</span>
                  </div>
                </div>
                <p className="muted" style={{ marginTop: 8 }}>
                  {bestRow.rationale}
                </p>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      setVizPlayType(bestRow.playType);
                      runViz(bestRow.playType);
                    }}
                    disabled={vizLoading}
                  >
                    <Icon name="map" /> {vizLoading ? "Generating…" : "Generate map"}
                  </button>

                  <button className="btn" type="button" onClick={() => authenticatedDownload(getPdfUrlForPlayType(bestRow.playType)).catch(e => setError(String(e)))}>
                    <Icon name="pdf" /> Export 1-page PDF
                  </button>
                </div>
              </div>
            ) : null}

            {/* Top-K bars */}
            <div className="barList" style={{ marginTop: 12 }}>
              {rows.map((r, idx) => {
                // Stable bar scale (avoid “blowing up” on high PPP)
                const width = clamp01(Number(r.pppPred) / 1.4) * 100;
                return (
                  <div key={`${r.playType}-${idx}`} className="barRow">
                    <div className="barLabel">
                      <div style={{ fontWeight: 900, fontSize: 12 }}>
                        #{idx + 1} {r.playType}
                      </div>
                      <div className="muted">
                        gap <span className="mono">{fmt(r.pppGap, 3)}</span>
                      </div>
                    </div>

                    <div className="meter" aria-label={`Predicted PPP bar for ${r.playType}`}>
                      <div style={{ width: `${width}%` }} />
                    </div>

                    <div className="mono">{fmt(r.pppPred, 3)}</div>
                  </div>
                );
              })}
            </div>

            {/* Court map (simple, coach-friendly) */}
            <div style={{ marginTop: 14 }}>
              <div className="sectionTitle">
                <h2>Court map</h2>
                <span className="badge">SportyPy</span>
              </div>

              <div className="grid2">
                <div>
                  <label>
                    Play type
                    <select
                      className="input"
                      value={vizPlayType}
                      onChange={(e) => setVizPlayType(e.target.value)}
                      disabled={!rows.length}
                    >
                      {rows.length ? (
                        rows.map((rr) => (
                          <option key={rr.playType} value={rr.playType}>
                            {rr.playType}
                          </option>
                        ))
                      ) : (
                        <option value="">Run baseline first…</option>
                      )}
                    </select>
                  </label>

                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
                    <button className="btn" type="button" onClick={() => runViz()} disabled={!vizReady || vizLoading}>
                      <Icon name="map" /> {vizLoading ? "Generating…" : "Generate map"}
                    </button>

                    <button
                      className="btn"
                      type="button"
                      disabled={!vizReady}
                      onClick={() => authenticatedDownload(getPdfUrlForPlayType(vizPlayType)).catch(e => setError(String(e)))}
                    >
                      <Icon name="pdf" /> Export PDF
                    </button>
                  </div>

                  {vizError ? <div className="error">{vizError}</div> : null}
                  {viz?.caption ? (
                    <p className="muted" style={{ marginTop: 10 }}>
                      {viz.caption}
                    </p>
                  ) : (
                    <p className="muted" style={{ marginTop: 10 }}>
                      Generate a map to attach a clean visual to your recommendation.
                    </p>
                  )}
                </div>

                <div>
                  {!viz ? (
                    <div
                      style={{
                        borderRadius: 14,
                        border: "1px dashed rgba(15,23,42,0.18)",
                        padding: 12,
                        background: "rgba(255,255,255,0.7)",
                      }}
                    >
                      <p className="muted" style={{ margin: 0 }}>
                        No map yet. Generate a map for the selected play type.
                      </p>
                    </div>
                  ) : (
                    <div className="imgFrame">
                      <img src={`data:image/png;base64,${viz.image_base64}`} alt="Court map" />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Details (kept for defense / trust, but not in the main coach flow) */}
            <details style={{ marginTop: 14 }}>
              <summary style={{ cursor: "pointer", fontWeight: 900, color: "rgba(15,23,42,0.9)" }}>
                Show details table
              </summary>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
                <button className="btn" type="button" onClick={() => setShowMath((v) => !v)}>
                  {showMath ? "Hide raw fields" : "Show raw fields"}
                </button>
              </div>

              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Play Type</th>
                      <th>Pred PPP</th>
                      <th>Our PPP (shrunk)</th>
                      <th>Opp Allowed (shrunk)</th>
                      <th>Gap</th>
                      {showMath ? (
                        <>
                          <th>Poss (our)</th>
                          <th>Poss% (our)</th>
                          <th>Rel (our)</th>
                          <th>Rel (opp)</th>
                          <th>League PPP (off)</th>
                          <th>League PPP (def)</th>
                        </>
                      ) : null}
                      <th>Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, idx) => {
                      const raw = r.raw ?? {};
                      return (
                        <tr key={`${r.playType}-${idx}`}>
                          <td>{idx + 1}</td>
                          <td style={{ fontWeight: 900 }}>{r.playType}</td>
                          <td>{fmt(r.pppPred, 3)}</td>
                          <td>{fmt(r.pppOff, 3)}</td>
                          <td>{fmt(r.pppDef, 3)}</td>
                          <td>{fmt(r.pppGap, 3)}</td>

                          {showMath ? (
                            <>
                              <td>{Number.isFinite(Number(raw.POSS_OFF)) ? fmtInt(raw.POSS_OFF) : "—"}</td>
                              <td>{safePct(raw.POSS_PCT_OFF)}</td>
                              <td>{fmt(raw.RELIABILITY_WEIGHT_OFF, 3)}</td>
                              <td>{fmt(raw.RELIABILITY_WEIGHT_DEF, 3)}</td>
                              <td>{fmt(raw.PPP_LEAGUE_OFF, 3)}</td>
                              <td>{fmt(raw.PPP_LEAGUE_DEF, 3)}</td>
                            </>
                          ) : null}

                          <td style={{ whiteSpace: "normal", minWidth: 360 }}>{r.rationale}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </div>

      {/* ✅ More info (all explanations live here, at the bottom) */}
      <details className="moreInfo">
        <summary>More info</summary>

        <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
          <div className="card" style={{ padding: 12 }}>
            <div className="sectionTitle" style={{ marginBottom: 6 }}>
              <h2>How the baseline works</h2>
              <span className="badge">Explainability</span>
            </div>

            <p className="muted" style={{ lineHeight: 1.55 }}>
              <span className="mono">
                {baselineInfo?.formula ?? "PPP_PRED = w_off * PPP_OFF_SHRUNK + w_def * PPP_DEF_SHRUNK"}
              </span>
            </p>

            <p className="muted" style={{ lineHeight: 1.55, marginTop: 8 }}>
              <strong>Why shrinkage?</strong>{" "}
              {baselineInfo?.whyShrinkage ??
                "It reduces small-sample noise so play types with few possessions don’t look unrealistically good or bad."}
            </p>
          </div>

          {baselineInfo?.definitions ? (
            <div className="card" style={{ padding: 12 }}>
              <div className="sectionTitle" style={{ marginBottom: 6 }}>
                <h2>Definitions</h2>
                <span className="badge">Terms</span>
              </div>

              <div style={{ display: "grid", gap: 8 }}>
                {Object.entries(baselineInfo.definitions).map(([kk, vv]) => (
                  <div key={kk} className="muted" style={{ lineHeight: 1.55 }}>
                    <span className="mono">{kk}</span>: {vv}
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="card" style={{ padding: 12 }}>
            <div className="sectionTitle" style={{ marginBottom: 6 }}>
              <h2>What happens when you run this</h2>
              <span className="badge">Backend</span>
            </div>

            <ul className="muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.65 }}>
              <li>Frontend sends: season, our team, opponent, k, and weights.</li>
              <li>Backend computes shrunk offense PPP (our) by play type.</li>
              <li>Backend computes shrunk defense-allowed PPP (opponent) by play type.</li>
              <li>Baseline prediction applies the weighted blend and sorts descending.</li>
              <li>Response includes Top-K rows + rationale + raw fields (optional).</li>
            </ul>
          </div>
        </div>
      </details>
    </section>
  );
}