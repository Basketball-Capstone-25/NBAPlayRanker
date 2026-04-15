// Model metrics page UI.

"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchModelMetrics, fetchPipelineInfo, fetchMlAnalysis } from "../../../services/analytics";

type MetricsRow = {
  model: string;
  RMSE_mean: number;
  RMSE_std: number;
  MAE_mean: number;
  MAE_std: number;
  R2_mean: number;
  R2_std: number;
};

type MetricsResponse = {
  n_splits: number;
  metrics: MetricsRow[];
  rf_vs_baseline_t: number | null;
  rf_vs_baseline_p: number | null;
};

type PipelineInfo = {
  dataSource?: string;
  cleaning_and_aggregation?: string[];
  modeling?: string[];
};

type MlTuning = {
  cv: string;
  features_used: string[];
  ridge: { best_params: { alpha: number }; best_rmse: number };
  random_forest: { best_params: any; best_rmse: number };
  gradient_boosting: { best_params: any; best_rmse: number };
};

type MlAnalysisSlim = {
  dataset?: {
    rows_after_filters: number;
    min_poss_filter: number;
    n_seasons: number;
    seasons: string[];
    n_teams?: number;
    n_play_types?: number;
    feature_cols?: string[];
    target_col?: string;
  };
  feature_selection?: {
    select_k_best?: { k: number; selected: string[] };
    rfe?: { selected: string[] };
    correlation_filter?: { threshold: number; kept: string[]; dropped: string[] };
  };
  model_selection?: {
    tuning: MlTuning;
  };
};

function fmt(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function fmtP(n: any) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  if (x < 0.0001) return "<0.0001";
  return x.toFixed(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function safeNum(n: any, fallback = NaN) {
  const x = Number(n);
  return Number.isFinite(x) ? x : fallback;
}

function prettyModelName(m: string) {
  // keep your backend labels, but soften common ones for display
  const s = String(m || "");
  if (/random\s*forest/i.test(s)) return "Random Forest";
  if (/gradient\s*boost/i.test(s)) return "Gradient Boosting";
  return s;
}

type TabKey = "scoreboard" | "holdout" | "tuning" | "trace";

export default function ModelMetricsPage() {
  // Data
  const [pipeline, setPipeline] = useState<PipelineInfo | null>(null);
  const [nSplits, setNSplits] = useState<number>(5);
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [ml, setMl] = useState<MlAnalysisSlim | null>(null);

  // UI state
  const [tab, setTab] = useState<TabKey>("scoreboard");
  const [compareA, setCompareA] = useState<string>(""); // “better” (left)
  const [compareB, setCompareB] = useState<string>(""); // “baseline/other” (right)
  const [poss, setPoss] = useState<number>(100); // possessions scale for “basketball terms”
  const [tuningRequested, setTuningRequested] = useState<boolean>(false);

  // Loading flags
  const [loadingPipeline, setLoadingPipeline] = useState<boolean>(false);
  const [loadingHoldout, setLoadingHoldout] = useState<boolean>(false);
  const [loadingTuning, setLoadingTuning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loading = loadingPipeline || loadingHoldout || loadingTuning;

  // Guards against Strict Mode duplicate calls + stale results
  const activeReqIdRef = useRef<number>(0);
  const lastPipelineKeyRef = useRef<string>("");
  const lastHoldoutKeyRef = useRef<string>("");
  const lastTuningKeyRef = useRef<string>("");

  async function loadPipeline(opts?: { force?: boolean }) {
    const key = "pipeline";
    if (!opts?.force && lastPipelineKeyRef.current === key && pipeline) return;
    lastPipelineKeyRef.current = key;

    const reqId = ++activeReqIdRef.current;

    try {
      setLoadingPipeline(true);
      const p = await fetchPipelineInfo();
      if (reqId !== activeReqIdRef.current) return;
      setPipeline(p);
    } catch (e) {
      // Pipeline not critical — don’t hard fail the page
      console.warn("[model-metrics] pipeline load failed:", e);
    } finally {
      if (reqId === activeReqIdRef.current) setLoadingPipeline(false);
    }
  }

  async function loadHoldout(opts?: { force?: boolean }) {
    const key = `holdout:nSplits=${nSplits}`;
    if (!opts?.force && lastHoldoutKeyRef.current === key && data) return;
    lastHoldoutKeyRef.current = key;

    const reqId = ++activeReqIdRef.current;

    try {
      setError(null);
      setLoadingHoldout(true);
      const m = await fetchModelMetrics(nSplits);
      if (reqId !== activeReqIdRef.current) return;
      setData(m);
    } catch (e: any) {
      if (reqId !== activeReqIdRef.current) return;
      console.error(e);
      setError(e?.message ?? "Failed to load holdout model metrics.");
      setData(null);
    } finally {
      if (reqId === activeReqIdRef.current) setLoadingHoldout(false);
    }
  }

  async function loadTuning(opts?: { force?: boolean }) {
    if (!tuningRequested && !opts?.force) return;

    const key = `tuning:nSplits=${nSplits}:minPoss=25:refresh=false`;
    if (!opts?.force && lastTuningKeyRef.current === key && ml) return;
    lastTuningKeyRef.current = key;

    const reqId = ++activeReqIdRef.current;

    try {
      setError(null);
      setLoadingTuning(true);
      const a = await fetchMlAnalysis({ nSplits, minPoss: 25, refresh: false });
      if (reqId !== activeReqIdRef.current) return;
      setMl(a);
    } catch (e: any) {
      if (reqId !== activeReqIdRef.current) return;
      console.error(e);
      setError(e?.message ?? "Failed to load tuning evidence.");
      setMl(null);
    } finally {
      if (reqId === activeReqIdRef.current) setLoadingTuning(false);
    }
  }

  async function refreshAll() {
    // Force reload everything. Tuning only reloads if it’s been requested (or already present).
    await loadPipeline({ force: true });
    await loadHoldout({ force: true });
    if (tuningRequested || ml) await loadTuning({ force: true });
  }

  // Initial + changes: keep page snappy (pipeline + holdout first)
  useEffect(() => {
    loadPipeline();
    loadHoldout();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadHoldout();
    if (tuningRequested) loadTuning();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nSplits]);

  // Lazy-load tuning when user opens the tab
  useEffect(() => {
    if (tab !== "tuning") return;
    if (!tuningRequested) setTuningRequested(true);
    loadTuning();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  const rows = useMemo(() => {
    return Array.isArray(data?.metrics) ? data!.metrics : [];
  }, [data]);

  const modelNames = useMemo(() => {
    return rows.map((r) => r.model);
  }, [rows]);

  const bestModel = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.RMSE_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => a.RMSE_mean - b.RMSE_mean)[0];
  }, [rows]);

  const worstModel = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.RMSE_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => b.RMSE_mean - a.RMSE_mean)[0];
  }, [rows]);

  const baselineGuess = useMemo(() => {
    // try to find a “Baseline” label if it exists; otherwise pick worst as the “comparison”
    const baseline = rows.find((r) => /baseline/i.test(r.model));
    return baseline ?? worstModel ?? null;
  }, [rows, worstModel]);

  // Default comparisons once we have data
  useEffect(() => {
    if (!rows.length) return;
    if (!compareA) {
      setCompareA(bestModel?.model ?? rows[0].model);
    }
    if (!compareB) {
      setCompareB(baselineGuess?.model ?? rows[Math.min(1, rows.length - 1)].model);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows.length]);

  // RMSE bars (normalized)
  const rmseBars = useMemo(() => {
    if (!rows.length) return [];
    const vals = rows.map((r) => safeNum(r.RMSE_mean)).filter((x) => Number.isFinite(x));
    if (!vals.length) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const denom = Math.max(1e-9, max - min);

    return rows.map((r) => {
      const v = safeNum(r.RMSE_mean);
      const score = Number.isFinite(v) ? 1 - (v - min) / denom : 0;
      return { model: r.model, widthPct: clamp01(score) * 100 };
    });
  }, [rows]);

  const holdoutSplitsLabel = useMemo(() => {
    const s = Number(data?.n_splits ?? nSplits);
    return Number.isFinite(s) ? s : nSplits;
  }, [data, nSplits]);

  const t = data?.rf_vs_baseline_t;
  const p = data?.rf_vs_baseline_p;

  const tuning = ml?.model_selection?.tuning;

  const bestTuned = useMemo(() => {
    if (!tuning) return null;
    const candidates = [
      { name: "Ridge", rmse: tuning.ridge?.best_rmse ?? Infinity },
      { name: "RandomForest", rmse: tuning.random_forest?.best_rmse ?? Infinity },
      { name: "GradientBoosting", rmse: tuning.gradient_boosting?.best_rmse ?? Infinity },
    ].filter((x) => Number.isFinite(x.rmse));
    if (!candidates.length) return null;
    return candidates.sort((a, b) => a.rmse - b.rmse)[0];
  }, [tuning]);

  const selectedFeatures = useMemo(() => {
    const fromTuning = tuning?.features_used;
    if (Array.isArray(fromTuning) && fromTuning.length) return fromTuning;

    const kbest = ml?.feature_selection?.select_k_best?.selected;
    if (Array.isArray(kbest) && kbest.length) return kbest;

    const rfe = ml?.feature_selection?.rfe?.selected;
    if (Array.isArray(rfe) && rfe.length) return rfe;

    return [];
  }, [tuning, ml]);

  // Comparison math (basketball terms)
  const compare = useMemo(() => {
    if (!rows.length || !compareA || !compareB) return null;
    const a = rows.find((r) => r.model === compareA);
    const b = rows.find((r) => r.model === compareB);
    if (!a || !b) return null;

    const aRMSE = safeNum(a.RMSE_mean);
    const bRMSE = safeNum(b.RMSE_mean);
    const aMAE = safeNum(a.MAE_mean);
    const bMAE = safeNum(b.MAE_mean);

    const dRMSE = bRMSE - aRMSE; // positive means A is better (lower error)
    const dMAE = bMAE - aMAE;

    const pctRMSE = Number.isFinite(bRMSE) && bRMSE > 0 ? (dRMSE / bRMSE) * 100 : NaN;
    const pctMAE = Number.isFinite(bMAE) && bMAE > 0 ? (dMAE / bMAE) * 100 : NaN;

    const possN = Math.max(1, Math.min(200, Number.isFinite(poss) ? poss : 100));
    const pointsPer100_RMSE = dRMSE * 100;
    const pointsPerGame_RMSE = dRMSE * possN;

    const pointsPer100_MAE = dMAE * 100;
    const pointsPerGame_MAE = dMAE * possN;

    return {
      a,
      b,
      dRMSE,
      dMAE,
      pctRMSE,
      pctMAE,
      possN,
      pointsPer100_RMSE,
      pointsPerGame_RMSE,
      pointsPer100_MAE,
      pointsPerGame_MAE,
    };
  }, [rows, compareA, compareB, poss]);

  const statusLine = useMemo(() => {
    if (error) return null;
    if (!loading) return null;

    if (loadingHoldout) return "Running holdout evaluation across seasons…";
    if (loadingPipeline) return "Loading pipeline summary…";
    if (loadingTuning) return "Loading tuning evidence (heavy)…";
    return "Loading…";
  }, [error, loading, loadingPipeline, loadingHoldout, loadingTuning]);

  const topBadge = useMemo(() => {
    if (!bestModel) return null;
    const rmse = safeNum(bestModel.RMSE_mean);
    return `Best holdout: ${prettyModelName(bestModel.model)} (RMSE ${fmt(rmse, 3)} PPP)`;
  }, [bestModel]);

  const pillStyle = (active: boolean) => ({
    padding: "8px 12px",
    borderRadius: 999,
    border: active ? "1px solid rgba(59,130,246,0.45)" : "1px solid rgba(15,23,42,0.10)",
    background: active ? "rgba(59,130,246,0.10)" : "rgba(255,255,255,0.8)",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: active ? 800 : 700,
    color: active ? "rgba(30,64,175,0.95)" : "rgba(15,23,42,0.85)",
    boxShadow: active ? "0 10px 25px rgba(59,130,246,0.12)" : "none",
    userSelect: "none" as const,
  });

  return (
    <section className="card" style={{ padding: 0, overflow: "hidden" }}>
      {/* HERO */}
      <div
        style={{
          padding: "18px 18px 16px",
          background:
            "radial-gradient(1200px 400px at 10% 10%, rgba(59,130,246,0.22), transparent 60%), radial-gradient(900px 340px at 90% 0%, rgba(16,185,129,0.16), transparent 55%), linear-gradient(180deg, rgba(15,23,42,0.06), rgba(15,23,42,0.02))",
          borderBottom: "1px solid rgba(15,23,42,0.08)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div style={{ maxWidth: 780 }}>
            <h1 className="h1" style={{ marginBottom: 6 }}>
              Model Performance
              <span style={{ opacity: 0.8, fontWeight: 700 }}> — Baseline vs ML</span>
            </h1>

            <p className="muted" style={{ fontSize: 14, margin: "6px 0 0" }}>
              Think of this like a <strong>scouting report</strong> for the model. We test generalization across
              seasons (holdout) and show tuning evidence (how we searched features + hyperparameters).
            </p>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              <span className="badge blue" style={{ background: "rgba(29, 66, 138, 0.12)", color: "rgba(29,66,138,0.95)" }}>
                PPP target
              </span>
              <span className="badge" style={{ background: "rgba(15, 23, 42, 0.06)" }}>
                Season holdout
              </span>
              <span className="badge" style={{ background: "rgba(16,185,129,0.10)", color: "rgba(5,150,105,0.95)" }}>
                Coach-friendly interpretation
              </span>
              {topBadge ? (
                <span className="badge blue" style={{ background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}>
                  {topBadge}
                </span>
              ) : null}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href="/context">
              Back: Context Simulator
            </Link>
            <Link className="btn" href="/statistical-analysis">
              Next: Statistical Analysis
            </Link>
          </div>
        </div>

        {/* Controls row */}
        <div
          style={{
            marginTop: 14,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ fontSize: 13, fontWeight: 700, color: "rgba(15,23,42,0.85)" }}>
              Season folds (n_splits)
              <select
                className="input"
                style={{ width: 110, display: "inline-block", marginLeft: 8 }}
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
                disabled={loadingHoldout || loadingTuning}
              >
                {[3, 4, 5, 6].map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </label>

            <button className="btn" type="button" onClick={refreshAll} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh"}
            </button>

            {statusLine ? (
              <span className="muted" style={{ fontSize: 12 }}>
                {statusLine}
              </span>
            ) : null}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <div style={pillStyle(tab === "scoreboard")} onClick={() => setTab("scoreboard")}>
              Scoreboard
            </div>
            <div style={pillStyle(tab === "holdout")} onClick={() => setTab("holdout")}>
              Holdout
            </div>
            <div style={pillStyle(tab === "tuning")} onClick={() => setTab("tuning")}>
              Tuning
            </div>
            <div style={pillStyle(tab === "trace")} onClick={() => setTab("trace")}>
              Traceability
            </div>
          </div>
        </div>

        {error ? (
          <div
            style={{
              marginTop: 12,
              padding: 12,
              borderRadius: 14,
              border: "1px solid rgba(239,68,68,0.22)",
              background: "rgba(239,68,68,0.06)",
              color: "rgba(153,27,27,0.95)",
              fontSize: 13,
              fontWeight: 700,
            }}
          >
            {error}
          </div>
        ) : null}
      </div>

      {/* BODY */}
      <div style={{ padding: 18 }}>
        {/* SCOREBOARD TAB */}
        {tab === "scoreboard" ? (
          <>
            {/* KPI strip */}
            <div className="grid" style={{ marginTop: 2 }}>
              <div className="kpi">
                <div className="label">What’s being predicted</div>
                <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                  Play Type PPP
                </div>
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  Points per possession by play type (team-level).
                </div>
              </div>

              <div className="kpi">
                <div className="label">Holdout method</div>
                <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                  Season splits
                </div>
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  Grouped by season to prevent leakage.
                </div>
              </div>

              <div className="kpi">
                <div className="label">Folds</div>
                <div className="value">{holdoutSplitsLabel}</div>
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  More folds = more repeated testing.
                </div>
              </div>
            </div>

            {/* Model comparison panel */}
            <div
              style={{
                marginTop: 14,
                borderRadius: 18,
                border: "1px solid rgba(15,23,42,0.10)",
                background:
                  "radial-gradient(900px 280px at 15% 10%, rgba(59,130,246,0.10), transparent 55%), radial-gradient(900px 280px at 85% 0%, rgba(16,185,129,0.10), transparent 55%), rgba(255,255,255,0.7)",
                padding: 14,
                boxShadow: "0 16px 30px rgba(15,23,42,0.06)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <div>
                  <h2 style={{ margin: "2px 0 6px", fontSize: 16, fontWeight: 900 }}>
                    “How much better is it?” (Basketball terms)
                  </h2>
                  <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                    RMSE/MAE are <strong>prediction error</strong> on PPP. Converting PPP error to “points”:
                    <br />
                    <span style={{ fontFamily: "var(--mono)" }}>
                      0.01 PPP error ≈ 1.0 point per 100 possessions (of prediction error)
                    </span>
                  </p>
                </div>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                  <label style={{ fontSize: 13, fontWeight: 800 }}>
                    Possessions scale
                    <input
                      className="input"
                      type="number"
                      min={50}
                      max={200}
                      step={1}
                      value={poss}
                      onChange={(e) => setPoss(Number(e.target.value))}
                      style={{ width: 110, marginLeft: 8 }}
                    />
                  </label>
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
                <label style={{ fontSize: 13, fontWeight: 800 }}>
                  Model A (better)
                  <select
                    className="input"
                    style={{ width: 220, marginLeft: 8 }}
                    value={compareA}
                    onChange={(e) => setCompareA(e.target.value)}
                    disabled={!modelNames.length}
                  >
                    {modelNames.map((m) => (
                      <option key={m} value={m}>
                        {prettyModelName(m)}
                      </option>
                    ))}
                  </select>
                </label>

                <label style={{ fontSize: 13, fontWeight: 800 }}>
                  Model B (compare)
                  <select
                    className="input"
                    style={{ width: 220, marginLeft: 8 }}
                    value={compareB}
                    onChange={(e) => setCompareB(e.target.value)}
                    disabled={!modelNames.length}
                  >
                    {modelNames.map((m) => (
                      <option key={m} value={m}>
                        {prettyModelName(m)}
                      </option>
                    ))}
                  </select>
                </label>

                <button
                  className="btn"
                  type="button"
                  onClick={() => {
                    // quick swap
                    const a = compareA;
                    setCompareA(compareB);
                    setCompareB(a);
                  }}
                  disabled={!compareA || !compareB}
                >
                  Swap
                </button>

                <button
                  className="btn"
                  type="button"
                  onClick={() => {
                    if (bestModel?.model) setCompareA(bestModel.model);
                    if (baselineGuess?.model) setCompareB(baselineGuess.model);
                  }}
                  disabled={!rows.length}
                >
                  Compare vs Baseline
                </button>
              </div>

              {/* Comparison output */}
              <div style={{ marginTop: 12 }}>
                {compare ? (
                  <div className="grid" style={{ marginTop: 0 }}>
                    <div className="kpi">
                      <div className="label">RMSE improvement (PPP error)</div>
                      <div className="value" style={{ fontSize: 18, fontWeight: 900 }}>
                        {compare.dRMSE >= 0 ? "+" : "—"}
                        {fmt(Math.abs(compare.dRMSE), 4)} PPP
                      </div>
                      <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                        {Number.isFinite(compare.pctRMSE)
                          ? `${fmt(Math.abs(compare.pctRMSE), 1)}% lower RMSE vs ${prettyModelName(compare.b.model)}`
                          : "Percent improvement unavailable."}
                      </div>
                    </div>

                    <div className="kpi">
                      <div className="label">≈ points / 100 possessions (error)</div>
                      <div className="value" style={{ fontSize: 18, fontWeight: 900 }}>
                        {compare.pointsPer100_RMSE >= 0 ? "+" : "—"}
                        {fmt(Math.abs(compare.pointsPer100_RMSE), 2)}
                      </div>
                      <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                        “Closer to true PPP” by ~{fmt(Math.abs(compare.pointsPer100_RMSE), 2)} points per 100 poss.
                      </div>
                    </div>

                    <div className="kpi">
                      <div className="label">≈ points / {compare.possN} possessions (error)</div>
                      <div className="value" style={{ fontSize: 18, fontWeight: 900 }}>
                        {compare.pointsPerGame_RMSE >= 0 ? "+" : "—"}
                        {fmt(Math.abs(compare.pointsPerGame_RMSE), 2)}
                      </div>
                      <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                        Scales the same error to your chosen possession count.
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                    Select two models to compare.
                  </p>
                )}

                <div
                  style={{
                    marginTop: 10,
                    padding: 12,
                    borderRadius: 14,
                    border: "1px solid rgba(15,23,42,0.10)",
                    background: "rgba(255,255,255,0.7)",
                  }}
                >
                  <div style={{ fontWeight: 900, fontSize: 13, marginBottom: 6 }}>Coach translation (quick)</div>
                  <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, margin: 0 }}>
                    <li>
                      <strong>PPP</strong> is points per possession for a play type (like “PnR Ball Handler”).
                    </li>
                    <li>
                      <strong>RMSE</strong> is “typical miss” on PPP predictions. Lower means the model is more reliable.
                    </li>
                    <li>
                      A drop of <strong>0.01 PPP</strong> in RMSE ≈ <strong>1 point per 100 possessions</strong> less
                      prediction error (not guaranteed points scored).
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Quick RMSE bar scan */}
            {rows.length ? (
              <div style={{ marginTop: 14 }}>
                <h2 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>
                  Holdout scan (lower RMSE is better)
                </h2>
                <p className="muted" style={{ fontSize: 13, marginTop: 0 }}>
                  Bars are scaled within this run. This is the fastest way for a coach to see what wins.
                </p>

                <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
                  {rmseBars.map((b) => {
                    const row = rows.find((r) => r.model === b.model);
                    const isBest = bestModel?.model === b.model;

                    return (
                      <div
                        key={b.model}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "220px 1fr 90px",
                          gap: 12,
                          alignItems: "center",
                        }}
                      >
                        <div style={{ fontSize: 13 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                            <strong title={`RMSE ${fmt(row?.RMSE_mean, 3)} ± ${fmt(row?.RMSE_std, 3)}`}>
                              {prettyModelName(b.model)}
                            </strong>
                            {isBest ? (
                              <span
                                className="badge blue"
                                style={{
                                  background: "rgba(59,130,246,0.12)",
                                  color: "rgba(30,64,175,0.95)",
                                }}
                              >
                                best
                              </span>
                            ) : null}
                          </div>
                          <div className="muted" style={{ fontSize: 11 }}>
                            RMSE {fmt(row?.RMSE_mean, 3)} ± {fmt(row?.RMSE_std, 3)}
                          </div>
                        </div>

                        <div
                          style={{
                            height: 12,
                            borderRadius: 999,
                            background: "rgba(15,23,42,0.08)",
                            overflow: "hidden",
                            boxShadow: "inset 0 1px 2px rgba(15,23,42,0.08)",
                          }}
                        >
                          <div
                            style={{
                              width: `${b.widthPct}%`,
                              height: "100%",
                              background: isBest
                                ? "linear-gradient(90deg, rgba(59,130,246,0.70), rgba(16,185,129,0.55))"
                                : "linear-gradient(90deg, rgba(15,23,42,0.35), rgba(15,23,42,0.20))",
                            }}
                          />
                        </div>

                        <div style={{ fontFamily: "var(--mono)", fontSize: 12, textAlign: "right" }}>
                          {fmt(row?.RMSE_mean, 3)}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <button className="btn" type="button" onClick={() => setTab("holdout")} disabled={!rows.length}>
                    Open full holdout table
                  </button>
                  <button className="btn" type="button" onClick={() => setTab("tuning")}>
                    Open tuning evidence
                  </button>
                </div>
              </div>
            ) : loadingHoldout ? (
              <div style={{ marginTop: 14 }}>
                <h2 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>Holdout scan</h2>
                <p className="muted" style={{ fontSize: 13, marginTop: 0 }}>
                  Running season holdout… (this step computes the table + bars)
                </p>
              </div>
            ) : null}

            {/* Stats line in plain terms */}
            {data ? (
              <div style={{ marginTop: 14 }}>
                <h2 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>Is the improvement real?</h2>
                <div
                  style={{
                    padding: 12,
                    borderRadius: 14,
                    border: "1px solid rgba(15,23,42,0.10)",
                    background: "rgba(255,255,255,0.75)",
                  }}
                >
                  <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                    Optional paired test (fold-by-fold RMSE): t = <strong>{fmt(t, 3)}</strong> | p ={" "}
                    <strong>{fmtP(p)}</strong>
                  </p>
                  <p className="muted" style={{ fontSize: 12, margin: "8px 0 0" }}>
                    Coach translation: smaller p-value means the “better model” wins more consistently across seasons,
                    not just by luck. If p is “—”, SciPy might not be installed or there weren’t enough folds.
                  </p>
                </div>
              </div>
            ) : null}
          </>
        ) : null}

        {/* HOLDOUT TAB */}
        {tab === "holdout" ? (
          <>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div>
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Holdout metrics table</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  This is the “generalizes to unseen seasons” scoreboard. Lower RMSE/MAE is better.
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("scoreboard")}>
                Back to Scoreboard
              </button>
            </div>

            {rows.length ? (
              <div style={{ marginTop: 12, overflowX: "auto" }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>RMSE (mean ± std)</th>
                      <th>MAE (mean ± std)</th>
                      <th>R² (mean ± std)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => {
                      const isBest = bestModel?.model === r.model;
                      return (
                        <tr key={r.model}>
                          <td>
                            <strong>{prettyModelName(r.model)}</strong>{" "}
                            {isBest ? (
                              <span className="badge blue" style={{ marginLeft: 8, background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}>
                                best
                              </span>
                            ) : null}
                          </td>
                          <td>
                            {fmt(r.RMSE_mean, 3)} ± {fmt(r.RMSE_std, 3)}
                          </td>
                          <td>
                            {fmt(r.MAE_mean, 3)} ± {fmt(r.MAE_std, 3)}
                          </td>
                          <td>
                            {fmt(r.R2_mean, 3)} ± {fmt(r.R2_std, 3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
                  <strong>Quick read:</strong> RMSE/MAE = prediction error on PPP. R² = how much variation the model explains.
                </div>
              </div>
            ) : loadingHoldout ? (
              <p className="muted" style={{ marginTop: 12 }}>
                Running holdout evaluation…
              </p>
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No holdout metrics loaded.
              </p>
            )}
          </>
        ) : null}

        {/* TUNING TAB */}
        {tab === "tuning" ? (
          <>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div>
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Tuning evidence</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  This shows we didn’t guess: we selected features and searched hyperparameters.
                </p>
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button className="btn" type="button" onClick={() => setTab("scoreboard")}>
                  Back to Scoreboard
                </button>
                <button className="btn" type="button" onClick={() => loadTuning({ force: true })} disabled={loadingTuning}>
                  {loadingTuning ? "Loading…" : "Reload tuning"}
                </button>
              </div>
            </div>

            {!tuningRequested ? (
              <div style={{ marginTop: 12 }}>
                <button className="btn" type="button" onClick={() => setTuningRequested(true)}>
                  Load tuning evidence
                </button>
                <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                  This step is heavy on first run. After it’s cached, it loads much faster.
                </p>
              </div>
            ) : null}

            {tuning ? (
              <div style={{ marginTop: 12 }}>
                <div className="grid">
                  <div className="kpi">
                    <div className="label">Tuning CV method</div>
                    <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                      {tuning.cv}
                    </div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      CV is grouped to avoid season leakage.
                    </div>
                  </div>

                  <div className="kpi">
                    <div className="label">Features used</div>
                    <div className="value">{selectedFeatures.length || "—"}</div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      Final feature set used for tuning.
                    </div>
                  </div>

                  <div className="kpi">
                    <div className="label">Best tuned model</div>
                    <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                      {bestTuned ? `${bestTuned.name} (${fmt(bestTuned.rmse, 4)})` : "—"}
                    </div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      Tuned RMSE is internal CV error (not holdout).
                    </div>
                  </div>
                </div>

                <div style={{ overflowX: "auto", marginTop: 12 }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Best RMSE (tuning)</th>
                        <th>Best Params</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>
                          <strong>Ridge</strong>
                        </td>
                        <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.ridge.best_rmse, 4)}</td>
                        <td style={{ fontFamily: "var(--mono)" }}>alpha={fmt(tuning.ridge.best_params.alpha, 3)}</td>
                      </tr>
                      <tr>
                        <td>
                          <strong>Random Forest</strong>
                        </td>
                        <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.random_forest.best_rmse, 4)}</td>
                        <td style={{ fontFamily: "var(--mono)" }}>
                          {JSON.stringify(tuning.random_forest.best_params)}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <strong>Gradient Boosting</strong>
                        </td>
                        <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.gradient_boosting.best_rmse, 4)}</td>
                        <td style={{ fontFamily: "var(--mono)" }}>
                          {JSON.stringify(tuning.gradient_boosting.best_params)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <details style={{ marginTop: 12 }}>
                  <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 900 }}>
                    Show feature list used for tuning
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
                      fontFamily: "var(--mono)",
                    }}
                  >
                    {JSON.stringify(selectedFeatures, null, 2)}
                  </pre>
                </details>

                <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
                  <strong>Coach translation:</strong> tuning is the “practice reps” where we try lots of settings; holdout
                  is the “real game” test on unseen seasons.
                </div>
              </div>
            ) : tuningRequested && loadingTuning ? (
              <p className="muted" style={{ marginTop: 12 }}>
                Loading tuning evidence…
              </p>
            ) : tuningRequested ? (
              <p className="muted" style={{ marginTop: 12 }}>
                Tuning evidence not loaded yet.
              </p>
            ) : null}
          </>
        ) : null}

        {/* TRACEABILITY TAB */}
        {tab === "trace" ? (
          <>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div>
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Traceability</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  For the committee: what ran where, and what data steps were applied.
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("scoreboard")}>
                Back to Scoreboard
              </button>
            </div>

            <div className="grid" style={{ marginTop: 12 }}>
              <div className="kpi">
                <div className="label">
                  <strong style={{ color: "rgba(15,23,42,0.9)" }}>Frontend (Next.js)</strong>
                </div>
                <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, marginTop: 10 }}>
                  <li>Data Explorer: view dataset + columns used</li>
                  <li>Matchup (Baseline): explainable baseline scoring</li>
                  <li>Context Simulator (AI): ML-based context adjustments</li>
                  <li>Model Performance: holdout metrics + interpretation</li>
                  <li>Statistical Analysis: EDA + selection + tuning evidence</li>
                </ul>
              </div>

              <div className="kpi">
                <div className="label">
                  <strong style={{ color: "rgba(15,23,42,0.9)" }}>Backend (FastAPI)</strong>
                </div>
                <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, marginTop: 10 }}>
                  <li>/metrics/baseline-vs-ml (holdout metrics per model)</li>
                  <li>/analysis/ml (EDA, feature selection, tuning)</li>
                  <li>/meta/pipeline (data source + preprocessing summary)</li>
                </ul>
              </div>

              <div className="kpi">
                <div className="label">
                  <strong style={{ color: "rgba(15,23,42,0.9)" }}>Data + preprocessing</strong>
                </div>
                <p className="muted" style={{ fontSize: 13, marginTop: 10 }}>
                  {pipeline?.dataSource
                    ? pipeline.dataSource
                    : loadingPipeline
                    ? "Loading pipeline summary…"
                    : "Data source not loaded (backend may be offline)."}
                </p>
                <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, marginTop: 8 }}>
                  {(pipeline?.cleaning_and_aggregation ?? [
                    "Aggregate player rows into team-level rows (possession-weighted).",
                    "Recompute team usage % to avoid double-counting (POSS_PCT).",
                    "Add reliability weights (shrinkage) to reduce noise from small sample sizes.",
                  ]).map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <h3 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>Defense </h3>
              <ol className="muted" style={{ fontSize: 13, paddingLeft: 18, margin: 0 }}>
                <li>We compared baseline vs ML — not one default model.</li>
                <li>We used season holdout so results generalize to unseen seasons.</li>
                <li>We show tuning evidence (features + params) for transparency.</li>
                <li>We explain improvements in basketball terms (PPP error → points per possessions).</li>
              </ol>
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}
