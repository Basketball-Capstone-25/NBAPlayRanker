// Shot statistical analysis page UI.
"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchShotMlAnalysis } from "../../../services/shotAnalysis";

type CorrMatrix = { labels: string[]; matrix: number[][] };

type TargetCorrRow = { feature: string; corr: number; abs: number };

type ShotMlAnalysisResponse = {
  dataset: {
    rows: number;
    n_games: number;
    n_seasons: number;
    feature_cols_numeric: string[];
    feature_cols_categorical: string[];
    target_col: string;
  };
  eda: {
    points: any;
    dist?: any;
    hist_points: { bins: number[]; counts: number[] };
    missing_counts: Record<string, number>;
  };
  correlations: CorrMatrix;
  target_feature_corr: TargetCorrRow[];
  feature_selection: {
    correlation_filter: { threshold: number; kept: string[]; dropped: string[] };
    select_k_best: { k: number; selected: string[]; scores: { feature: string; score: number }[] };
    rfe: { selected: string[]; ranking: { feature: string; rank: number }[] };
  };
  model_selection: {
    tuning: {
      cv: string;
      features_used: string[];
      ridge: { best_params: { alpha: number }; best_rmse: number };
      random_forest: { best_params: any; best_rmse: number };
      gradient_boosting: { best_params: any; best_rmse: number };
    };
  };
};

/** Pretty-print numeric values consistently */
function fmt(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

/** Integer formatting for counts */
function fmtInt(n: any) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return String(Math.round(x));
}

/** Clamp to [0,1] (handy for bar widths / alpha intensities) */
function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

/**
 * Heatmap cell background for correlation matrix:
 * - positive corr => blue-ish
 * - negative corr => red-ish
 * - magnitude controls alpha (stronger correlation => darker cell)
 */
function corrBg(v: number) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "transparent";
  const a = clamp01(Math.abs(x));
  if (x >= 0) return `rgba(29, 66, 138, ${0.10 + 0.35 * a})`;
  return `rgba(200, 16, 46, ${0.10 + 0.35 * a})`;
}

/** LocalStorage helpers (safe for environments where storage might be blocked) */
function safeLocalGet(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function safeLocalSet(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore (privacy mode / blocked storage)
  }
}

/** CSV export for target-feature correlation table (useful for reporting / appendix) */
function toCsvTargetCorr(rows: TargetCorrRow[]) {
  const header = ["feature", "corr", "abs"];
  const lines = [header.join(",")];
  for (const r of rows) {
    const row = [r.feature, String(r.corr), String(r.abs)];
    lines.push(row.map((x) => `"${String(x).replaceAll('"', '""')}"`).join(","));
  }
  return lines.join("\n");
}

/**
 * MiniHistogram
 * -------------
 * A tiny histogram renderer that uses div bars.
 * The backend provides:
 *  - bins: either edges (len = counts+1) or centers (len = counts)
 *  - counts: frequencies per bin
 *
 * We compute per-bar height relative to max count so it’s readable.
 */
function MiniHistogram({
  title,
  hist,
}: {
  title: string;
  hist: { bins: number[]; counts: number[] };
}) {
  const bins = hist?.bins ?? [];
  const counts = hist?.counts ?? [];
  const max = Math.max(1, ...counts); // prevent divide-by-zero

  /**
   * Bin label logic: derives a range from edges or centers.
   */
  function getBinRange(i: number): { lo: number; hi: number } {
    const edges = bins;

    // Common case: edges array
    if (edges.length === counts.length + 1) {
      return { lo: Number(edges[i]), hi: Number(edges[i + 1]) };
    }

    // Alternate case: bins represent centers
    if (edges.length === counts.length && edges.length > 0) {
      const c = Number(edges[i]);
      const prev = i > 0 ? Number(edges[i - 1]) : c;
      const next = i < edges.length - 1 ? Number(edges[i + 1]) : c;
      const halfLeft = (c - prev) / 2;
      const halfRight = (next - c) / 2;
      return { lo: c - halfLeft, hi: c + halfRight };
    }

    // Fallback (should be rare)
    return { lo: NaN, hi: NaN };
  }

  return (
    <div className="kpi">
      <div className="label" style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
        <strong style={{ color: "rgba(15,23,42,0.9)" }}>{title}</strong>
        <span className="badge">bins: {counts.length}</span>
      </div>

      {/* Bar chart container */}
      <div style={{ marginTop: 10, display: "flex", gap: 4, alignItems: "flex-end", height: 70 }}>
        {counts.map((c, i) => {
          // bar height scaled relative to maximum count
          const h = Math.round((c / max) * 70);
          const { lo, hi } = getBinRange(i);
          return (
            <div
              key={i}
              title={`${fmt(lo, 2)}–${fmt(hi, 2)} / count ${fmtInt(c)}`}
              style={{
                width: 10,
                height: h,
                borderRadius: 6,
                background: "rgba(15,23,42,0.25)",
              }}
            />
          );
        })}
      </div>

      <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
        Hover bars to see: <strong>bin range</strong> / <strong>count</strong>.
      </div>
    </div>
  );
}

export default function ShotStatisticalAnalysisPage() {
  /**
   * nSplits controls GroupKFold (grouped by GAME_ID on the backend).
   * Higher splits = more CV folds (slower but potentially more stable estimates).
   */
  const [nSplits, setNSplits] = useState(5);

  // Main analysis payload from backend
  const [data, setData] = useState<ShotMlAnalysisResponse | null>(null);

  // Request state
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Status UX (small “Updated ✅” badge, etc.)
  const [statusHint, setStatusHint] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshEverySec, setRefreshEverySec] = useState<number>(120);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  // User controls for exploring output
  const [featureQuery, setFeatureQuery] = useState<string>("");
  const [topN, setTopN] = useState<number>(10);
  const [heatmapMaxCols, setHeatmapMaxCols] = useState<number>(18);
  const [showRaw, setShowRaw] = useState<boolean>(false);

  // Prevents stale responses when params change rapidly.
  const requestIdRef = useRef(0);

  // didInitRef ensures we only restore LocalStorage settings once
  const didInitRef = useRef(false);

  // Restore prefs once (keeps UX consistent across reloads)
  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;

    const raw = safeLocalGet("nbaPlayRanker_shotStatAnalysis_v2");
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw);

      // Basic preference parsing with safe defaults
      const ns = Number(parsed.nSplits);
      const ar = Boolean(parsed.autoRefresh);
      const re = Number(parsed.refreshEverySec);
      const tq = String(parsed.featureQuery ?? "");
      const tn = Number(parsed.topN);
      const hm = Number(parsed.heatmapMaxCols);
      const sr = Boolean(parsed.showRaw);

      // Clamp/validate to avoid broken UI due to corrupted storage values
      if ([3, 4, 5, 6].includes(ns)) setNSplits(ns);
      if (Number.isFinite(re) && re >= 15 && re <= 300) setRefreshEverySec(re);
      setAutoRefresh(ar);

      if (Number.isFinite(tn) && tn >= 5 && tn <= 25) setTopN(tn);
      if (Number.isFinite(hm) && hm >= 8 && hm <= 40) setHeatmapMaxCols(hm);

      setFeatureQuery(tq);
      setShowRaw(sr);
    } catch {
      // ignore parse errors (treat as “no saved prefs”)
    }
  }, []);

  // Persist prefs whenever relevant state changes
  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotStatAnalysis_v2",
      JSON.stringify({
        nSplits,
        autoRefresh,
        refreshEverySec,
        featureQuery,
        topN,
        heatmapMaxCols,
        showRaw,
      })
    );
  }, [nSplits, autoRefresh, refreshEverySec, featureQuery, topN, heatmapMaxCols, showRaw]);

  /**
   * Load analysis from backend.
   * - refresh=false: typical mode (uses backend cache if available)
   * - refresh=true: forces backend recomputation (heavier, slower)
   * - silent=true: keep UI stable (don’t clear results / don’t spam hints)
   */
  async function load({
    refresh = false,
    silent = false,
  }: { refresh?: boolean; silent?: boolean } = {}) {
    const myId = ++requestIdRef.current;

    try {
      setLoading(true);
      setErr(null);

      if (!silent) setStatusHint(refresh ? "Recomputing analysis…" : "Loading analysis…");

      // Backend call (utils wraps URL + query params)
      const res = (await fetchShotMlAnalysis({ nSplits, refresh })) as ShotMlAnalysisResponse;

      // If a newer request started after ours, ignore this result
      if (requestIdRef.current !== myId) return;

      setData(res);
      setLastUpdated(Date.now());
      setStatusHint("Updated ✅");
      window.setTimeout(() => setStatusHint(""), 900);
    } catch (e: any) {
      if (requestIdRef.current !== myId) return;

      console.error(e);
      setErr(e?.message ?? "Failed to load shot statistical analysis.");
      setData(null);
      setStatusHint("Load failed");
      window.setTimeout(() => setStatusHint(""), 1200);
    } finally {
      if (requestIdRef.current === myId) setLoading(false);
    }
  }

  // Load on mount + whenever nSplits changes
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nSplits]);

  // Optional auto-refresh (pulls cached results without forcing recomputation)
  useEffect(() => {
    if (!autoRefresh) return;

    const t = window.setInterval(() => {
      load({ refresh: false, silent: true });
    }, Math.max(15, refreshEverySec) * 1000);

    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, refreshEverySec, nSplits]);

  // Shorthand to the model tuning block (best params + best RMSE for each model)
  const tuning = data?.model_selection?.tuning;

  // Determine best model by lowest RMSE
  const bestModel = useMemo(() => {
    if (!tuning) return null;

    const candidates = [
      { name: "Ridge", rmse: tuning.ridge?.best_rmse ?? Infinity },
      { name: "RandomForest", rmse: tuning.random_forest?.best_rmse ?? Infinity },
      { name: "GradientBoosting", rmse: tuning.gradient_boosting?.best_rmse ?? Infinity },
    ].filter((x) => Number.isFinite(x.rmse));

    if (!candidates.length) return null;
    return candidates.sort((a, b) => a.rmse - b.rmse)[0];
  }, [tuning]);

  /**
   * Filterable + sorted target-feature correlations.
   * - featureQuery narrows the list by substring match
   * - always sort by abs corr descending (most influential features first)
   */
  const targetCorrFiltered = useMemo(() => {
    const rows = Array.isArray(data?.target_feature_corr) ? data!.target_feature_corr : [];
    const q = featureQuery.trim().toLowerCase();
    const filtered = q ? rows.filter((r) => r.feature.toLowerCase().includes(q)) : rows;
    return [...filtered].sort((a, b) => b.abs - a.abs);
  }, [data, featureQuery]);

  /**
   * Top-N slice for the “bar list” view.
   * NOTE: the original version tried to guard topN with a weird clamp01 expression.
   * Here we keep the logic as-is but use a clearer fallback: slice(0, topN).
   */
  const topTargetCorr = useMemo(() => {
    const safeN = Number.isFinite(topN) && topN > 0 ? topN : 10;
    return targetCorrFiltered.slice(0, safeN);
  }, [targetCorrFiltered, topN]);

  /**
   * Correlation heatmap is potentially large, so we limit to the first N labels/cols.
   * This keeps DOM size manageable and avoids slow rendering.
   */
  const corrLabels = useMemo(() => {
    const labels = data?.correlations?.labels ?? [];
    const max = Math.max(1, heatmapMaxCols);
    return labels.slice(0, max);
  }, [data, heatmapMaxCols]);

  const corrMatrix = useMemo(() => {
    const m = data?.correlations?.matrix ?? [];
    const max = Math.max(1, heatmapMaxCols);
    return m.slice(0, max).map((row) => (Array.isArray(row) ? row.slice(0, max) : []));
  }, [data, heatmapMaxCols]);

  // Simple “vibrant” hero panel styling to match the rest of your app
  const heroStyle: React.CSSProperties = {
    borderRadius: 18,
    padding: "18px 18px 14px",
    background:
      "linear-gradient(135deg, rgba(56,189,248,0.16), rgba(99,102,241,0.14), rgba(34,197,94,0.10))",
    border: "1px solid rgba(255,255,255,0.10)",
  };

  return (
    <main className="page" style={{ paddingBottom: 56 }}>
      {/* Header: title + nav + controls */}
      <header className="page__header" style={heroStyle}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <h1 className="h1" style={{ margin: 0 }}>
              Shot Statistical Analysis
            </h1>
            <p className="muted" style={{ fontSize: 14, marginTop: 6, marginBottom: 0 }}>
              EDA, correlation heatmap, feature selection, and model tuning for the Shot Intelligence dataset.
            </p>
          </div>

          {/* Simple “guided flow” nav between pages */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn btn--secondary" href="/shot-model-metrics">
              Back: Shot Model Metrics
            </Link>
            <Link className="btn" href="/shot-plan">
              Next: Shot Plan
            </Link>
          </div>
        </div>

        {/* Controls row */}
        <div
          style={{
            marginTop: 12,
            display: "flex",
            gap: 10,
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            {/* n_splits changes the backend CV configuration */}
            <label style={{ fontSize: 13 }}>
              n_splits (GroupKFold by GAME_ID)
              <select
                className="input"
                style={{ width: 140, display: "inline-block", marginLeft: 8 }}
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
              >
                {[3, 4, 5, 6].map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </label>

            {/* Refresh = fetch cached/most recent analysis */}
            <button
              className="btn"
              type="button"
              onClick={() => load({ refresh: false })}
              disabled={loading}
            >
              {loading ? "Loading…" : "Refresh"}
            </button>

            {/* Recompute = force backend recomputation (slow) */}
            <button
              className="btn btn--secondary"
              type="button"
              onClick={() => load({ refresh: true })}
              disabled={loading}
              title="Forces backend recomputation (slower)"
            >
              {loading ? "Working…" : "Recompute"}
            </button>

            {/* Auto-refresh = periodic background refresh */}
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>

            {/* Auto-refresh frequency */}
            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Every
              <select
                className="input"
                style={{ width: 110 }}
                value={refreshEverySec}
                onChange={(e) => setRefreshEverySec(Number(e.target.value))}
                disabled={!autoRefresh}
              >
                {[15, 30, 60, 120, 300].map((s) => (
                  <option key={s} value={s}>
                    {s}s
                  </option>
                ))}
              </select>
            </label>

            {/* Short-lived status badge (Updated / Loading / etc.) */}
            {statusHint ? <span className="badge">{statusHint}</span> : null}
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            {/* Highlight best model so the page “tells a story” quickly */}
            {bestModel ? (
              <span className="badge blue">
                Best (lowest RMSE): {bestModel.name} ({fmt(bestModel.rmse, 3)})
              </span>
            ) : (
              <span className="badge">Best model: —</span>
            )}

            {/* Timestamp is helpful during demos / defense (shows it’s live) */}
            {lastUpdated ? (
              <span className="muted" style={{ fontSize: 12 }}>
                Updated: {new Date(lastUpdated).toLocaleTimeString()}
              </span>
            ) : null}

            {/* Debug/transparency toggle */}
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={showRaw}
                onChange={(e) => setShowRaw(e.target.checked)}
              />
              Show raw JSON
            </label>
          </div>
        </div>
      </header>

      {/* Error block (kept separate so the rest of the page can still render) */}
      {err ? (
        <section className="card" style={{ marginTop: 14 }}>
          <p className="muted" style={{ marginTop: 0 }}>
            {err}
          </p>
        </section>
      ) : null}

      {/* Dataset snapshot: quick metadata so the analysis feels grounded */}
      {data?.dataset ? (
        <section className="card" style={{ marginTop: 14 }}>
          <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>Dataset snapshot</h2>
          <div className="grid" style={{ marginTop: 10 }}>
            <div className="kpi">
              <div className="label">Rows</div>
              <div className="value">{fmtInt(data.dataset.rows)}</div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                Shot attempts after cleaning.
              </div>
            </div>
            <div className="kpi">
              <div className="label">Games</div>
              <div className="value">{fmtInt(data.dataset.n_games)}</div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                CV groups = games (leakage-safe).
              </div>
            </div>
            <div className="kpi">
              <div className="label">Seasons</div>
              <div className="value">{fmtInt(data.dataset.n_seasons)}</div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                Dataset2 seasons coverage.
              </div>
            </div>
            <div className="kpi">
              <div className="label">Target</div>
              <div className="value" style={{ fontFamily: "var(--mono)", fontSize: 16 }}>
                {data.dataset.target_col}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                Model predicts expected points.
              </div>
            </div>
          </div>

          <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <span className="badge">
              Numeric features: {data.dataset.feature_cols_numeric?.length ?? 0}
            </span>
            <span className="badge">
              Categorical features: {data.dataset.feature_cols_categorical?.length ?? 0}
            </span>
            {tuning?.cv ? <span className="badge blue">CV: {tuning.cv}</span> : null}
          </div>
        </section>
      ) : null}

      {/* EDA: distribution + missingness */}
      {data?.eda ? (
        <section className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>EDA</h2>
            <span className="muted" style={{ fontSize: 13 }}>
              Quick quality checks (distribution + missingness).
            </span>
          </div>

          <div className="grid" style={{ marginTop: 10 }}>
            {/* Tiny histogram for target distribution (points/EP) */}
            <MiniHistogram title="Points distribution" hist={data.eda.hist_points} />

            {/* Missingness summary (top 10 columns) */}
            <div className="kpi">
              <div className="label">
                <strong style={{ color: "rgba(15,23,42,0.9)" }}>Missing values</strong>
              </div>
              <div style={{ marginTop: 10, display: "grid", gap: 6 }}>
                {Object.entries(data.eda.missing_counts ?? {})
                  .sort((a, b) => Number(b[1]) - Number(a[1]))
                  .slice(0, 10)
                  .map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 10,
                        fontSize: 13,
                      }}
                    >
                      <span style={{ fontFamily: "var(--mono)" }}>{k}</span>
                      <span className="badge">{fmtInt(v)}</span>
                    </div>
                  ))}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                Top 10 columns by missing count.
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {/* Correlation heatmap table */}
      {data?.correlations?.labels?.length ? (
        <section className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>
              Correlation heatmap (numeric features + target)
            </h2>

            {/* Limit columns so the UI stays fast */}
            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Max cols
              <select
                className="input"
                style={{ width: 110 }}
                value={heatmapMaxCols}
                onChange={(e) => setHeatmapMaxCols(Number(e.target.value))}
              >
                {[12, 14, 18, 22, 30, 40].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
            Showing first <strong>{corrLabels.length}</strong> columns for readability/performance.
          </div>

          <div style={{ overflowX: "auto", marginTop: 10 }}>
            <table className="table">
              <thead>
                <tr>
                  {/* Empty corner header for row labels */}
                  <th style={{ position: "sticky", left: 0, background: "rgba(15,23,42,0.03)" }} />
                  {corrLabels.map((c) => (
                    <th key={c} style={{ fontFamily: "var(--mono)" }}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {corrLabels.map((rLabel, i) => (
                  <tr key={rLabel}>
                    {/* Sticky row label so it stays visible while horizontally scrolling */}
                    <td
                      style={{
                        position: "sticky",
                        left: 0,
                        background: "rgba(255,255,255,0.92)",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {rLabel}
                    </td>

                    {(corrMatrix[i] ?? []).map((v, j) => (
                      <td
                        key={`${i}-${j}`}
                        style={{
                          background: corrBg(v),
                          fontFamily: "var(--mono)",
                          textAlign: "right",
                        }}
                        title={`corr(${rLabel}, ${corrLabels[j]}) = ${fmt(v, 3)}`}
                      >
                        {fmt(v, 2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {/* Target-feature correlations (ranked by magnitude) */}
      {Array.isArray(data?.target_feature_corr) && data!.target_feature_corr.length ? (
        <section className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>
              Target-feature correlations (top by |corr|)
            </h2>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              {/* TopN control changes how many bars we show */}
              <label style={{ fontSize: 13 }}>
                Top N
                <select
                  className="input"
                  style={{ width: 100, display: "inline-block", marginLeft: 8 }}
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                >
                  {[5, 10, 15, 20, 25].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>

              {/* Export full filtered table for appendix/reporting */}
              <button
                className="btn btn--secondary"
                type="button"
                onClick={() => {
                  const csv = toCsvTargetCorr(targetCorrFiltered);
                  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `shot_target_corr_splits_${nSplits}.csv`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Export CSV
              </button>
            </div>
          </div>

          {/* Feature search */}
          <div style={{ marginTop: 10, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <input
              className="input"
              style={{ width: 320, maxWidth: "100%" }}
              placeholder="Filter features… (e.g. distance, angle, shot_clock)"
              value={featureQuery}
              onChange={(e) => setFeatureQuery(e.target.value)}
            />
            <span className="muted" style={{ fontSize: 13 }}>
              Showing <strong>{Math.min(topTargetCorr.length, topN)}</strong> of{" "}
              <strong>{targetCorrFiltered.length}</strong> matches.
            </span>
          </div>

          {/* Bar list (simple, readable ranking) */}
          <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
            {topTargetCorr.map((r) => {
              const w = clamp01(r.abs) * 100; // magnitude controls width
              const pos = r.corr >= 0; // sign controls color direction
              return (
                <div
                  key={r.feature}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "260px 1fr 80px",
                    gap: 10,
                    alignItems: "center",
                  }}
                >
                  <div style={{ fontSize: 13, fontFamily: "var(--mono)" }}>{r.feature}</div>

                  <div
                    style={{
                      height: 10,
                      borderRadius: 999,
                      background: "rgba(15,23,42,0.08)",
                      overflow: "hidden",
                      border: "1px solid rgba(255,255,255,0.10)",
                    }}
                    title={`corr = ${fmt(r.corr, 3)} | abs = ${fmt(r.abs, 3)}`}
                  >
                    <div
                      style={{
                        width: `${w}%`,
                        height: "100%",
                        background: pos
                          ? "linear-gradient(90deg, rgba(59,130,246,0.65), rgba(34,197,94,0.35))"
                          : "linear-gradient(90deg, rgba(239,68,68,0.55), rgba(245,158,11,0.35))",
                      }}
                    />
                  </div>

                  <div style={{ textAlign: "right", fontFamily: "var(--mono)", fontSize: 12 }}>
                    {fmt(r.corr, 3)}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      ) : null}

      {/* Feature selection summary (as returned by backend) */}
      {data?.feature_selection ? (
        <section className="card">
          <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>Feature selection (numeric)</h2>

          <div className="grid" style={{ marginTop: 10 }}>
            {/* Correlation filter (removes redundant variables) */}
            <div className="kpi">
              <div className="label">
                <strong style={{ color: "rgba(15,23,42,0.9)" }}>Correlation filter</strong>
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                Threshold: <strong>{fmt(data.feature_selection.correlation_filter.threshold, 2)}</strong>
              </div>
              <div style={{ marginTop: 10 }}>
                <span className="badge blue">
                  Kept: {data.feature_selection.correlation_filter.kept.length}
                </span>{" "}
                <span className="badge" style={{ marginLeft: 8 }}>
                  Dropped: {data.feature_selection.correlation_filter.dropped.length}
                </span>
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                Removes highly redundant numeric features.
              </div>
            </div>

            {/* SelectKBest (univariate scoring) */}
            <div className="kpi">
              <div className="label">
                <strong style={{ color: "rgba(15,23,42,0.9)" }}>SelectKBest</strong>
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                k = <strong>{data.feature_selection.select_k_best.k}</strong>
              </div>
              <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 8 }}>
                {data.feature_selection.select_k_best.selected.slice(0, 12).map((f) => (
                  <span key={f} className="badge blue">
                    {f}
                  </span>
                ))}
              </div>
              {data.feature_selection.select_k_best.selected.length > 12 ? (
                <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                  +{data.feature_selection.select_k_best.selected.length - 12} more…
                </div>
              ) : null}
            </div>

            {/* RFE (wrapper selection, using Ridge) */}
            <div className="kpi">
              <div className="label">
                <strong style={{ color: "rgba(15,23,42,0.9)" }}>RFE (Ridge)</strong>
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                Selected features:
              </div>
              <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 8 }}>
                {data.feature_selection.rfe.selected.slice(0, 12).map((f) => (
                  <span key={f} className="badge red">
                    {f}
                  </span>
                ))}
              </div>
              {data.feature_selection.rfe.selected.length > 12 ? (
                <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                  +{data.feature_selection.rfe.selected.length - 12} more…
                </div>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      {/* Model selection summary (tuned RMSE per model) */}
      {tuning ? (
        <section className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>Model selection summary</h2>

            {/* Copy-to-clipboard is useful for quick reporting / defense notes */}
            <button
              className="btn btn--secondary"
              type="button"
              onClick={async () => {
                if (!tuning) return;
                const best = bestModel ? `${bestModel.name} (${fmt(bestModel.rmse, 3)})` : "—";
                const text = `Shot ML tuning | CV=${tuning.cv} | features=${tuning.features_used?.length ?? 0} | best RMSE=${best}`;
                try {
                  await navigator.clipboard.writeText(text);
                  setStatusHint("Copied ✅");
                  window.setTimeout(() => setStatusHint(""), 900);
                } catch {
                  setStatusHint("Copy failed");
                  window.setTimeout(() => setStatusHint(""), 900);
                }
              }}
            >
              Copy summary
            </button>
          </div>

          <p className="muted" style={{ fontSize: 13, marginTop: 4 }}>
            CV method: <strong>{tuning.cv}</strong>. Models are tuned using RMSE (lower is better).
          </p>

          {/* Simple table so parameters are transparent */}
          <div style={{ overflowX: "auto", marginTop: 10 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Best RMSE</th>
                  <th>Best Params</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <strong>Ridge</strong>
                  </td>
                  <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.ridge.best_rmse, 4)}</td>
                  <td style={{ fontFamily: "var(--mono)" }}>
                    alpha={fmt(tuning.ridge.best_params.alpha, 3)}
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong>RandomForest</strong>
                  </td>
                  <td style={{ fontFamily: "var(--mono)" }}>
                    {fmt(tuning.random_forest.best_rmse, 4)}
                  </td>
                  <td style={{ fontFamily: "var(--mono)" }}>
                    {tuning.random_forest?.best_params
                      ? JSON.stringify(tuning.random_forest.best_params)
                      : "—"}
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong>GradientBoosting</strong>
                  </td>
                  <td style={{ fontFamily: "var(--mono)" }}>
                    {fmt(tuning.gradient_boosting.best_rmse, 4)}
                  </td>
                  <td style={{ fontFamily: "var(--mono)" }}>
                    {tuning.gradient_boosting?.best_params
                      ? JSON.stringify(tuning.gradient_boosting.best_params)
                      : "—"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {/* Raw payload (debug + transparency for evaluators) */}
      {showRaw && data ? (
        <section className="card">
          <details open>
            <summary style={{ cursor: "pointer", fontSize: 13 }}>
              Raw analysis JSON (debug / transparency)
            </summary>
            <pre
              style={{
                marginTop: 10,
                background: "rgba(15, 23, 42, 0.04)",
                padding: 12,
                borderRadius: 12,
                overflowX: "auto",
                fontSize: 12,
                border: "1px solid rgba(15, 23, 42, 0.08)",
              }}
            >
              {JSON.stringify(data, null, 2)}
            </pre>
          </details>
        </section>
      ) : null}
    </main>
  );
}
