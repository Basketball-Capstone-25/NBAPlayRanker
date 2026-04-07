// Statistical analysis page UI.
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchMlAnalysis } from "../../../services/analytics";

type CorrMatrix = { labels: string[]; matrix: number[][] };

type TargetCorrRow = { feature: string; corr: number; abs: number };

type MlAnalysisResponse = {
  dataset: {
    rows_after_filters: number;
    min_poss_filter: number;
    n_seasons: number;
    seasons: string[];
    n_teams: number;
    n_play_types: number;
    feature_cols: string[];
    target_col: string;
  };
  eda: {
    poss: any;
    ppp: any;
    hist_poss: { bins: number[]; counts: number[] };
    hist_ppp: { bins: number[]; counts: number[] };
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

function safeNum(n: any, fallback = NaN) {
  const x = Number(n);
  return Number.isFinite(x) ? x : fallback;
}

// ✅ Defensive array normalizer (prevents .length/.map crashes when backend omits arrays)
function asArray<T = any>(v: any): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

/**
 * Correlation heatmap cell background:
 * - positive => blue
 * - negative => red
 * - magnitude controls alpha
 */
function corrBg(v: number) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "transparent";
  const a = clamp01(Math.abs(x));
  if (x >= 0) return `rgba(59,130,246, ${0.10 + 0.35 * a})`; // vibrant blue
  return `rgba(239,68,68, ${0.10 + 0.35 * a})`; // vibrant red
}

type TabKey = "overview" | "eda" | "correlations" | "selection" | "models" | "raw";

/** ===== session cache (fast load without backend) ===== */
const CACHE_PREFIX = "statAnalysis:v1:";

function makeCacheKey(nSplits: number, minPoss: number) {
  return `${CACHE_PREFIX}nSplits=${nSplits}&minPoss=${minPoss}`;
}

type CachePayload = {
  ts: number;
  nSplits: number;
  minPoss: number;
  data: MlAnalysisResponse;
};

function readCache(nSplits: number, minPoss: number): CachePayload | null {
  try {
    const key = makeCacheKey(nSplits, minPoss);
    const raw = sessionStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachePayload;
    if (!parsed?.data?.dataset?.target_col) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeCache(payload: CachePayload) {
  try {
    const key = makeCacheKey(payload.nSplits, payload.minPoss);
    sessionStorage.setItem(key, JSON.stringify(payload));
  } catch {
    // ignore quota / serialization failures
  }
}

function clearCache(nSplits: number, minPoss: number) {
  try {
    sessionStorage.removeItem(makeCacheKey(nSplits, minPoss));
  } catch {
    // ignore
  }
}

/**
 * MiniHistogram
 * -------------
 * Adds two perf/UX upgrades:
 * - Compresses very large histograms down to <= 48 bars (grouped sums)
 * - Slightly more “product” visual with soft gradient
 */
function MiniHistogram({
  title,
  hist,
}: {
  title: string;
  hist: { bins: number[]; counts: number[] };
}) {
  const rawBins = hist?.bins ?? [];
  const rawCounts = hist?.counts ?? [];

  const digits = title.toLowerCase().includes("poss") ? 0 : 3;

  function compressHist(bins: number[], counts: number[], maxBars = 48) {
    if (!Array.isArray(counts) || counts.length <= maxBars) {
      return { bins, counts };
    }

    const stride = Math.ceil(counts.length / maxBars);
    const groupedCounts: number[] = [];
    const isEdges = bins.length === counts.length + 1;

    if (isEdges) {
      const edges = bins;
      const groupedEdges: number[] = [];
      groupedEdges.push(edges[0]);

      for (let start = 0; start < counts.length; start += stride) {
        const end = Math.min(counts.length, start + stride);
        let sum = 0;
        for (let i = start; i < end; i++) sum += Number.isFinite(counts[i]) ? counts[i] : 0;
        groupedCounts.push(sum);

        // push the edge at end
        groupedEdges.push(edges[end]);
      }
      return { bins: groupedEdges, counts: groupedCounts };
    }

    // centers: approximate grouped centers by average
    const centers = bins.length === counts.length ? bins : counts.map((_, i) => i);
    const groupedCenters: number[] = [];

    for (let start = 0; start < counts.length; start += stride) {
      const end = Math.min(counts.length, start + stride);
      let sum = 0;
      let cSum = 0;
      let cN = 0;

      for (let i = start; i < end; i++) {
        sum += Number.isFinite(counts[i]) ? counts[i] : 0;
        if (Number.isFinite(centers[i])) {
          cSum += centers[i];
          cN += 1;
        }
      }

      groupedCounts.push(sum);
      groupedCenters.push(cN ? cSum / cN : start);
    }

    return { bins: groupedCenters, counts: groupedCounts };
  }

  const { bins, counts } = compressHist(rawBins, rawCounts);

  const max = Math.max(1, ...counts);
  const total = counts.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);

  function getBinRange(i: number): { lo: number; hi: number } {
    // edges
    if (bins.length === counts.length + 1) {
      const lo = Number(bins[i]);
      const hi = Number(bins[i + 1]);
      return { lo, hi };
    }

    // centers
    if (bins.length === counts.length && bins.length > 0) {
      const c = Number(bins[i]);
      const prev = i > 0 ? Number(bins[i - 1]) : c;
      const next = i < bins.length - 1 ? Number(bins[i + 1]) : c;

      const halfLeft = (c - prev) / 2;
      const halfRight = (next - c) / 2;

      const lo = c - (Number.isFinite(halfLeft) ? halfLeft : 0);
      const hi = c + (Number.isFinite(halfRight) ? halfRight : 0);
      return { lo, hi };
    }

    return { lo: NaN, hi: NaN };
  }

  function rangeLabel(i: number) {
    const { lo, hi } = getBinRange(i);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return "—";
    const left = digits === 0 ? fmtInt(lo) : fmt(lo, digits);
    const right = digits === 0 ? fmtInt(hi) : fmt(hi, digits);
    return `${left}–${right}`;
  }

  const axisMin =
    bins.length === counts.length + 1 ? bins[0] : bins.length ? getBinRange(0).lo : NaN;

  const axisMax =
    bins.length === counts.length + 1
      ? bins[bins.length - 1]
      : bins.length
      ? getBinRange(Math.max(0, counts.length - 1)).hi
      : NaN;

  return (
    <div className="kpi">
      <div className="label" style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
        <strong style={{ color: "rgba(15,23,42,0.92)" }}>{title}</strong>
        <span className="badge">bins: {counts.length}</span>
      </div>

      <div style={{ marginTop: 10, display: "flex", gap: 4, alignItems: "flex-end", height: 74 }}>
        {counts.map((c, i) => {
          const h = Math.round((c / max) * 74);
          const r = rangeLabel(i);

          return (
            <div
              key={i}
              title={`${r} / count ${fmtInt(c)}`}
              style={{
                width: 10,
                height: h,
                borderRadius: 8,
                background: "linear-gradient(180deg, rgba(59,130,246,0.70), rgba(16,185,129,0.55))",
                opacity: 0.88,
              }}
            />
          );
        })}
      </div>

      <div
        className="muted"
        style={{
          marginTop: 8,
          fontSize: 12,
          display: "flex",
          justifyContent: "space-between",
          gap: 10,
        }}
      >
        <span>
          X:{" "}
          <strong style={{ fontFamily: "var(--mono)" }}>
            {Number.isFinite(axisMin)
              ? digits === 0
                ? fmtInt(axisMin)
                : fmt(axisMin, digits)
              : "—"}
          </strong>{" "}
          →{" "}
          <strong style={{ fontFamily: "var(--mono)" }}>
            {Number.isFinite(axisMax)
              ? digits === 0
                ? fmtInt(axisMax)
                : fmt(axisMax, digits)
              : "—"}
          </strong>
        </span>
        <span>
          Y max: <strong style={{ fontFamily: "var(--mono)" }}>{fmtInt(max)}</strong>
        </span>
      </div>

      <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
        Hover bars to see: <strong>bin range</strong> / <strong>count</strong>.
      </div>

      <details style={{ marginTop: 10 }}>
        <summary style={{ cursor: "pointer", fontSize: 13 }}>Show bin table (x range / y count)</summary>

        <div style={{ overflowX: "auto", marginTop: 10 }}>
          <table className="table">
            <thead>
              <tr>
                <th>Bin</th>
                <th>X range</th>
                <th>Count</th>
                <th>%</th>
              </tr>
            </thead>
            <tbody>
              {counts.map((c, i) => {
                const { lo, hi } = getBinRange(i);
                const left = Number.isFinite(lo) ? (digits === 0 ? fmtInt(lo) : fmt(lo, digits)) : "—";
                const right = Number.isFinite(hi) ? (digits === 0 ? fmtInt(hi) : fmt(hi, digits)) : "—";
                const pct = total > 0 ? (Number(c) / total) * 100 : 0;

                return (
                  <tr key={i}>
                    <td style={{ fontFamily: "var(--mono)" }}>{i + 1}</td>
                    <td style={{ fontFamily: "var(--mono)" }}>
                      {left}–{right}
                    </td>
                    <td style={{ fontFamily: "var(--mono)" }}>{fmtInt(c)}</td>
                    <td style={{ fontFamily: "var(--mono)" }}>{fmt(pct, 1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function TabPill({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: "8px 12px",
        borderRadius: 999,
        border: active ? "1px solid rgba(59,130,246,0.45)" : "1px solid rgba(15,23,42,0.10)",
        background: active ? "rgba(59,130,246,0.10)" : "rgba(255,255,255,0.82)",
        cursor: "pointer",
        fontSize: 13,
        fontWeight: active ? 900 : 800,
        color: active ? "rgba(30,64,175,0.95)" : "rgba(15,23,42,0.85)",
        boxShadow: active ? "0 14px 28px rgba(59,130,246,0.10)" : "none",
        userSelect: "none",
      }}
    >
      {label}
    </div>
  );
}

export default function StatisticalAnalysisPage() {
  const [tab, setTab] = useState<TabKey>("overview");

  // Controls (manual run to avoid repeated expensive backend calls)
  const [nSplits, setNSplits] = useState(5);
  const [minPoss, setMinPoss] = useState(25);

  // Data
  const [data, setData] = useState<MlAnalysisResponse | null>(null);

  // Cache meta (for “instant load”)
  const [cacheTs, setCacheTs] = useState<number | null>(null);

  // Request state
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Ignore stale responses
  const reqIdRef = useRef(0);

  function loadFromCacheIfPresent() {
    const cached = readCache(nSplits, minPoss);
    if (cached?.data) {
      setData(cached.data);
      setCacheTs(cached.ts);
      setErr(null);
      return true;
    }
    setCacheTs(null);
    return false;
  }

  async function runAnalysis(opts?: { refresh?: boolean }) {
    const refresh = Boolean(opts?.refresh);

    const reqId = ++reqIdRef.current;

    try {
      setLoading(true);
      setStage(refresh ? "Forcing recompute (backend)…" : "Fetching analysis (backend)…");
      setErr(null);

      const res = (await fetchMlAnalysis({
        nSplits,
        minPoss,
        refresh,
      })) as unknown as MlAnalysisResponse;

      if (reqId !== reqIdRef.current) return;

      setData(res);
      const payload: CachePayload = { ts: Date.now(), nSplits, minPoss, data: res };
      writeCache(payload);
      setCacheTs(payload.ts);

      setStage("Rendering dashboard…");
      // tiny defer so the UI can paint the result state smoothly
      setTimeout(() => {
        if (reqId === reqIdRef.current) setStage(null);
      }, 250);
    } catch (e: any) {
      if (reqId !== reqIdRef.current) return;

      console.error(e);
      setErr(e?.message ?? "Failed to load statistical analysis (backend /analysis/ml).");

      // If we have cached data, keep it on screen (don’t nuke UX).
      const hasCache = loadFromCacheIfPresent();
      if (!hasCache) setData(null);
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }

  // First mount: instant cache if available, otherwise run once
  useEffect(() => {
    const hadCache = loadFromCacheIfPresent();
    if (!hadCache) runAnalysis({ refresh: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When params change: show cached version instantly if it exists, otherwise prompt user to Run
  useEffect(() => {
    if (loading) return;
    loadFromCacheIfPresent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nSplits, minPoss]);

  const topTargetCorr = useMemo(() => {
    const rows = data?.target_feature_corr ?? [];
    return [...rows].sort((a, b) => b.abs - a.abs).slice(0, 10);
  }, [data]);

  const tuning = data?.model_selection?.tuning;

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

  // ✅ Safe accessors for Feature Selection tab (fixes dropped/kept/selected possibly missing)
  const corrFilter = data?.feature_selection?.correlation_filter;
  const corrKept = asArray<string>(corrFilter?.kept);
  const corrDropped = asArray<string>(corrFilter?.dropped);
  const kbestSelected = asArray<string>(data?.feature_selection?.select_k_best?.selected);
  const rfeSelected = asArray<string>(data?.feature_selection?.rfe?.selected);

  // Correlation heatmap mode (compact is MUCH lighter to render)
  const [heatmapMode, setHeatmapMode] = useState<"compact" | "full">("compact");
  const [compactK, setCompactK] = useState<number>(16);

  const heatmap = useMemo(() => {
    const corr = data?.correlations;
    if (!corr?.labels?.length || !corr?.matrix?.length) return null;

    if (heatmapMode === "full") return corr;

    // compact: target + top-K-1 correlated features
    const target = data?.dataset?.target_col;
    const top = (data?.target_feature_corr ?? [])
      .slice()
      .sort((a, b) => b.abs - a.abs)
      .slice(0, Math.max(1, compactK - 1))
      .map((x) => x.feature);

    const keep = new Set<string>();
    if (target) keep.add(target);
    for (const f of top) keep.add(f);

    // preserve original order for readability
    const labels = corr.labels.filter((l) => keep.has(l));
    const idxMap = new Map<string, number>();
    corr.labels.forEach((l, i) => idxMap.set(l, i));
    const idxs = labels.map((l) => idxMap.get(l) ?? -1).filter((i) => i >= 0);

    const matrix = idxs.map((ri) => idxs.map((ci) => corr.matrix?.[ri]?.[ci] ?? NaN));
    return { labels, matrix };
  }, [data, heatmapMode, compactK]);

  const cacheLabel = useMemo(() => {
    if (!cacheTs) return null;
    const d = new Date(cacheTs);
    return d.toLocaleString();
  }, [cacheTs]);

  // “Coach readable” takeaways
  const takeaways = useMemo(() => {
    if (!data) return null;

    const target = data.dataset?.target_col ?? "PPP";
    const rows = data.target_feature_corr ?? [];
    const top3 = rows.slice().sort((a, b) => b.abs - a.abs).slice(0, 3).map((x) => x.feature);

    const missing = data.eda?.missing_counts ?? {};
    const missingTotal = Object.values(missing).reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);

    const minPossFilter = data.dataset?.min_poss_filter ?? minPoss;
    const nSeasons = data.dataset?.n_seasons ?? data.dataset?.seasons?.length ?? 0;

    return {
      target,
      top3,
      missingTotal,
      minPossFilter,
      nSeasons,
    };
  }, [data, minPoss]);

  return (
    <section className="card" style={{ padding: 0, overflow: "hidden" }}>
      {/* HERO */}
      <div
        style={{
          padding: "18px 18px 16px",
          background:
            "radial-gradient(1200px 420px at 10% 10%, rgba(59,130,246,0.22), transparent 60%), radial-gradient(900px 360px at 90% 0%, rgba(16,185,129,0.16), transparent 55%), linear-gradient(180deg, rgba(15,23,42,0.06), rgba(15,23,42,0.02))",
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
          <div style={{ maxWidth: 820 }}>
            <h1 className="h1" style={{ marginBottom: 6 }}>
              Statistical Analysis
              <span style={{ opacity: 0.8, fontWeight: 700 }}> — show your work</span>
            </h1>

            <p className="muted" style={{ fontSize: 14, margin: 0 }}>
              Everything on this page comes from a single backend payload. “Defense dashboard” for:
              distributions, correlations, feature selection, and model tuning.
            </p>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              <span
                className="badge blue"
                style={{ background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}
              >
                Dataset1 (Synergy)
              </span>
              <span className="badge" style={{ background: "rgba(15, 23, 42, 0.06)" }}>
                Single endpoint: /analysis/ml
              </span>
              {bestModel ? (
                <span
                  className="badge blue"
                  style={{
                    background: "rgba(16,185,129,0.12)",
                    color: "rgba(5,150,105,0.95)",
                  }}
                >
                  Best tuned: {bestModel.name} (RMSE {fmt(bestModel.rmse, 3)})
                </span>
              ) : null}
              {cacheLabel ? (
                <span className="badge" style={{ background: "rgba(255,255,255,0.65)" }}>
                  Cached: {cacheLabel}
                </span>
              ) : null}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href="/model-metrics">
              Back: Model Performance
            </Link>
            <Link className="btn" href="/glossary">
              Next: Glossary
            </Link>
          </div>
        </div>

        {/* Controls */}
        <div
          style={{
            marginTop: 14,
            display: "flex",
            gap: 10,
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ fontSize: 13, fontWeight: 800, color: "rgba(15,23,42,0.85)" }}>
              n_splits
              <select
                className="input"
                style={{ width: 110, display: "inline-block", marginLeft: 8 }}
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
                disabled={loading}
              >
                {[3, 4, 5, 6].map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ fontSize: 13, fontWeight: 800, color: "rgba(15,23,42,0.85)" }}>
              min_poss
              <input
                className="input"
                style={{ width: 110, display: "inline-block", marginLeft: 8 }}
                type="number"
                value={minPoss}
                min={0}
                max={200}
                onChange={(e) => setMinPoss(Number(e.target.value))}
                disabled={loading}
              />
            </label>

            <button className="btn" type="button" onClick={() => runAnalysis({ refresh: false })} disabled={loading}>
              {loading ? "Running…" : "Run"}
            </button>

            <button
              className="btn"
              type="button"
              onClick={() => runAnalysis({ refresh: true })}
              disabled={loading}
              title="Forces backend to recompute (slow). Use only if you changed the backend logic or caches."
            >
              Force recompute
            </button>

            <button
              className="btn"
              type="button"
              onClick={() => {
                clearCache(nSplits, minPoss);
                setCacheTs(null);
              }}
              disabled={loading}
              title="Clears cached payload for the current settings."
            >
              Clear cache
            </button>

            {stage ? (
              <span className="muted" style={{ fontSize: 12 }}>
                {stage}
              </span>
            ) : null}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <TabPill active={tab === "overview"} label="Overview" onClick={() => setTab("overview")} />
            <TabPill active={tab === "eda"} label="EDA" onClick={() => setTab("eda")} />
            <TabPill active={tab === "correlations"} label="Correlations" onClick={() => setTab("correlations")} />
            <TabPill active={tab === "selection"} label="Feature Selection" onClick={() => setTab("selection")} />
            <TabPill active={tab === "models"} label="Model Tuning" onClick={() => setTab("models")} />
            <TabPill active={tab === "raw"} label="Raw JSON" onClick={() => setTab("raw")} />
          </div>
        </div>

        {err ? (
          <div
            style={{
              marginTop: 12,
              padding: 12,
              borderRadius: 14,
              border: "1px solid rgba(239,68,68,0.22)",
              background: "rgba(239,68,68,0.06)",
              color: "rgba(153,27,27,0.95)",
              fontSize: 13,
              fontWeight: 800,
            }}
          >
            {err}
          </div>
        ) : null}
      </div>

      {/* BODY */}
      <div style={{ padding: 18 }}>
        {/* If no data and not loading: callout */}
        {!data && !loading ? (
          <div
            style={{
              padding: 14,
              borderRadius: 16,
              border: "1px solid rgba(15,23,42,0.10)",
              background: "rgba(255,255,255,0.75)",
              boxShadow: "0 18px 30px rgba(15,23,42,0.05)",
            }}
          >
            <div style={{ fontWeight: 900, marginBottom: 6 }}>No results loaded for these settings.</div>
            <div className="muted" style={{ fontSize: 13 }}>
              If the backend is slow on first run, use <strong>Run</strong> once (it should cache after). Next time,
              this page will load instantly from cache.
            </div>
          </div>
        ) : null}

        {/* OVERVIEW */}
        {tab === "overview" ? (
          <>
            {data?.dataset ? (
              <div className="grid" style={{ marginTop: 0 }}>
                <div className="kpi">
                  <div className="label">Rows after filters</div>
                  <div className="value">{fmtInt(data.dataset.rows_after_filters)}</div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    Filter: POSS ≥ {fmtInt(data.dataset.min_poss_filter)}
                  </div>
                </div>

                <div className="kpi">
                  <div className="label">Coverage</div>
                  <div className="value">{fmtInt(data.dataset.n_seasons)} seasons</div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    Teams: {fmtInt(data.dataset.n_teams)} · Play types: {fmtInt(data.dataset.n_play_types)}
                  </div>
                </div>

                <div className="kpi">
                  <div className="label">Target</div>
                  <div className="value" style={{ fontFamily: "var(--mono)", fontSize: 14 }}>
                    {data.dataset.target_col}
                  </div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    Features: {fmtInt(data.dataset.feature_cols?.length ?? 0)}
                  </div>
                </div>

                <div className="kpi">
                  <div className="label">Best tuned model</div>
                  <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                    {bestModel ? `${bestModel.name}` : "—"}
                  </div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    Picked by lowest tuned RMSE
                  </div>
                </div>
              </div>
            ) : null}

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
              <h2 style={{ margin: "2px 0 8px", fontSize: 16, fontWeight: 900 }}>How to read this page</h2>
              <ol className="muted" style={{ fontSize: 13, paddingLeft: 18, margin: 0 }}>
                <li>
                  <strong>EDA</strong> shows distribution + missing values (data quality & filters).
                </li>
                <li>
                  <strong>Correlations</strong> explains relationships + redundancy (why we prune features).
                </li>
                <li>
                  <strong>Feature selection</strong> proves we didn’t guess (filter + KBest + RFE).
                </li>
                <li>
                  <strong>Model tuning</strong> compares candidate models (lowest RMSE wins).
                </li>
              </ol>

              {takeaways ? (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontWeight: 900, fontSize: 13, marginBottom: 6 }}>Quick takeaways (coach version)</div>
                  <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, margin: 0 }}>
                    <li>
                      Target is <strong>{takeaways.target}</strong> (points per possession for a play type).
                    </li>
                    <li>
                      Reliability filter: <strong>POSS ≥ {fmtInt(takeaways.minPossFilter)}</strong> to reduce small-sample
                      noise.
                    </li>
                    <li>
                      Top correlated features:{" "}
                      <strong>{takeaways.top3?.length ? takeaways.top3.join(", ") : "—"}</strong>
                    </li>
                    <li>
                      Missingness after filters: <strong>{fmtInt(takeaways.missingTotal)}</strong> total missing cells (across
                      columns).
                    </li>
                  </ul>
                </div>
              ) : null}

              <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button className="btn" type="button" onClick={() => setTab("eda")} disabled={!data}>
                  Go to EDA
                </button>
                <button className="btn" type="button" onClick={() => setTab("correlations")} disabled={!data}>
                  Go to Correlations
                </button>
                <button className="btn" type="button" onClick={() => setTab("models")} disabled={!data}>
                  Go to Model Tuning
                </button>
              </div>
            </div>
          </>
        ) : null}

        {/* EDA */}
        {tab === "eda" ? (
          <>
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
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>EDA</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  Distributions + missingness so reviewers can trust the dataset quality and filters.
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("overview")}>
                Back to Overview
              </button>
            </div>

            {data?.eda ? (
              <div className="grid" style={{ marginTop: 12 }}>
                <MiniHistogram title="POSS distribution" hist={data.eda.hist_poss} />
                <MiniHistogram title={`${data.dataset.target_col} distribution`} hist={data.eda.hist_ppp} />

                <div className="kpi">
                  <div className="label">
                    <strong style={{ color: "rgba(15,23,42,0.9)" }}>Missing values</strong>
                  </div>

                  <div style={{ marginTop: 10, display: "grid", gap: 6 }}>
                    {Object.entries(data.eda.missing_counts ?? {})
                      .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
                      .slice(0, 12)
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
                    Top 12 columns by missing count (after filters).
                  </div>
                </div>
              </div>
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No EDA loaded.
              </p>
            )}
          </>
        ) : null}

        {/* CORRELATIONS */}
        {tab === "correlations" ? (
          <>
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
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Correlations</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  Compact by default (faster). Blue = positive, red = negative.
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("overview")}>
                Back to Overview
              </button>
            </div>

            {topTargetCorr.length ? (
              <div style={{ marginTop: 12 }}>
                <h3 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>Top correlations with target</h3>
                <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
                  {topTargetCorr.map((r) => {
                    const w = clamp01(r.abs) * 100;
                    return (
                      <div
                        key={r.feature}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "240px 1fr 70px",
                          gap: 10,
                          alignItems: "center",
                        }}
                      >
                        <div style={{ fontSize: 13, fontFamily: "var(--mono)" }}>{r.feature}</div>

                        <div
                          style={{
                            height: 12,
                            borderRadius: 999,
                            background: "rgba(15,23,42,0.08)",
                            overflow: "hidden",
                            boxShadow: "inset 0 1px 2px rgba(15,23,42,0.08)",
                          }}
                          title={`corr = ${fmt(r.corr, 3)} | abs = ${fmt(r.abs, 3)}`}
                        >
                          <div
                            style={{
                              width: `${w}%`,
                              height: "100%",
                              background:
                                r.corr >= 0
                                  ? "linear-gradient(90deg, rgba(59,130,246,0.70), rgba(59,130,246,0.35))"
                                  : "linear-gradient(90deg, rgba(239,68,68,0.70), rgba(239,68,68,0.35))",
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
              </div>
            ) : null}

            {heatmap ? (
              <div style={{ marginTop: 16 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 12,
                    flexWrap: "wrap",
                    alignItems: "center",
                  }}
                >
                  <div>
                    <h3 style={{ margin: "8px 0 6px", fontSize: 16, fontWeight: 900 }}>
                      Correlation heatmap ({heatmapMode})
                    </h3>
                    <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                      Tip: keep it on <strong>Compact</strong> for speed; use <strong>Full</strong> only when needed.
                    </p>
                  </div>

                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                    <label style={{ fontSize: 13, fontWeight: 800 }}>
                      Mode
                      <select
                        className="input"
                        style={{ width: 140, marginLeft: 8 }}
                        value={heatmapMode}
                        onChange={(e) => setHeatmapMode(e.target.value as any)}
                      >
                        <option value="compact">Compact</option>
                        <option value="full">Full</option>
                      </select>
                    </label>

                    {heatmapMode === "compact" ? (
                      <label style={{ fontSize: 13, fontWeight: 800 }}>
                        Top-K
                        <input
                          className="input"
                          style={{ width: 90, marginLeft: 8 }}
                          type="number"
                          min={8}
                          max={32}
                          value={compactK}
                          onChange={(e) => setCompactK(Number(e.target.value))}
                        />
                      </label>
                    ) : null}
                  </div>
                </div>

                <div style={{ overflowX: "auto", marginTop: 10 }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th
                          style={{
                            position: "sticky",
                            left: 0,
                            background: "rgba(15,23,42,0.03)",
                          }}
                        >
                          {" "}
                        </th>
                        {heatmap.labels.map((c) => (
                          <th key={c} style={{ fontFamily: "var(--mono)" }}>
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {heatmap.labels.map((rLabel, i) => (
                        <tr key={rLabel}>
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

                          {(heatmap.matrix[i] ?? []).map((v, j) => (
                            <td
                              key={`${i}-${j}`}
                              style={{
                                background: corrBg(v),
                                fontFamily: "var(--mono)",
                                textAlign: "right",
                              }}
                              title={`corr(${rLabel}, ${heatmap.labels[j]}) = ${fmt(v, 3)}`}
                            >
                              {fmt(v, 2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No correlation matrix loaded.
              </p>
            )}
          </>
        ) : null}

        {/* FEATURE SELECTION */}
        {tab === "selection" ? (
          <>
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
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Feature Selection</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  Shows we reduced redundancy and picked features systematically (not vibes).
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("overview")}>
                Back to Overview
              </button>
            </div>

            {data?.feature_selection ? (
              <div className="grid" style={{ marginTop: 12 }}>
                <div className="kpi">
                  <div className="label">
                    <strong style={{ color: "rgba(15,23,42,0.9)" }}>Correlation filter</strong>
                  </div>
                  <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                    Threshold: <strong>{fmt(corrFilter?.threshold, 2)}</strong>
                  </div>

                  <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
                    <span
                      className="badge blue"
                      style={{ background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}
                    >
                      Kept: {corrKept.length}
                    </span>
                    <span className="badge" style={{ background: "rgba(15,23,42,0.06)" }}>
                      Dropped: {corrDropped.length}
                    </span>
                  </div>

                  <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                    Why: reduces multicollinearity and makes the model more stable.
                  </div>

                  <details style={{ marginTop: 10 }}>
                    <summary style={{ cursor: "pointer", fontSize: 13 }}>Show kept/dropped lists</summary>
                    <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                      <div>
                        <div style={{ fontWeight: 900, fontSize: 13, marginBottom: 6 }}>Kept</div>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                          {corrKept.map((f) => (
                            <span
                              key={f}
                              className="badge blue"
                              style={{ background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}
                            >
                              {f}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontWeight: 900, fontSize: 13, marginBottom: 6 }}>Dropped</div>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                          {corrDropped.map((f) => (
                            <span key={f} className="badge" style={{ background: "rgba(15,23,42,0.06)" }}>
                              {f}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </details>
                </div>

                <div className="kpi">
                  <div className="label">
                    <strong style={{ color: "rgba(15,23,42,0.9)" }}>SelectKBest (f_regression)</strong>
                  </div>
                  <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                    k = <strong>{data.feature_selection.select_k_best.k}</strong>
                  </div>

                  <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {kbestSelected.map((f) => (
                      <span
                        key={f}
                        className="badge blue"
                        style={{ background: "rgba(59,130,246,0.12)", color: "rgba(30,64,175,0.95)" }}
                      >
                        {f}
                      </span>
                    ))}
                  </div>

                  <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                    Why: fast univariate baseline that ranks features by signal.
                  </div>
                </div>

                <div className="kpi">
                  <div className="label">
                    <strong style={{ color: "rgba(15,23,42,0.9)" }}>RFE (Ridge wrapper)</strong>
                  </div>

                  <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {rfeSelected.map((f) => (
                      <span
                        key={f}
                        className="badge"
                        style={{ background: "rgba(16,185,129,0.12)", color: "rgba(5,150,105,0.95)" }}
                      >
                        {f}
                      </span>
                    ))}
                  </div>

                  <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                    Why: wrapper method picks features that improve the model the most.
                  </div>
                </div>
              </div>
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No feature selection results loaded.
              </p>
            )}
          </>
        ) : null}

        {/* MODEL TUNING */}
        {tab === "models" ? (
          <>
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
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Model Tuning</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  Shows hyperparameter search and the winner by RMSE (lower is better).
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("overview")}>
                Back to Overview
              </button>
            </div>

            {tuning ? (
              <div style={{ marginTop: 12 }}>
                <div className="grid">
                  <div className="kpi">
                    <div className="label">CV Method</div>
                    <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                      {tuning.cv}
                    </div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      Grouped by season to avoid leakage.
                    </div>
                  </div>

                  <div className="kpi">
                    <div className="label">Features used</div>
                    <div className="value">{fmtInt(tuning.features_used?.length ?? 0)}</div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      Final feature set used for tuning.
                    </div>
                  </div>

                  <div className="kpi">
                    <div className="label">Winner</div>
                    <div className="value" style={{ fontSize: 16, fontWeight: 900 }}>
                      {bestModel ? bestModel.name : "—"}
                    </div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      Lowest RMSE among tuned candidates.
                    </div>
                  </div>
                </div>

                <div style={{ overflowX: "auto", marginTop: 12 }}>
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
                        <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.random_forest.best_rmse, 4)}</td>
                        <td style={{ fontFamily: "var(--mono)" }}>
                          {tuning.random_forest?.best_params ? JSON.stringify(tuning.random_forest.best_params) : "—"}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <strong>GradientBoosting</strong>
                        </td>
                        <td style={{ fontFamily: "var(--mono)" }}>{fmt(tuning.gradient_boosting.best_rmse, 4)}</td>
                        <td style={{ fontFamily: "var(--mono)" }}>
                          {tuning.gradient_boosting?.best_params
                            ? JSON.stringify(tuning.gradient_boosting.best_params)
                            : "—"}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <details style={{ marginTop: 12 }}>
                  <summary style={{ cursor: "pointer", fontSize: 13 }}>Show features used for tuning</summary>
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
                    {JSON.stringify(tuning.features_used ?? [], null, 2)}
                  </pre>
                </details>

                <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                  Interpretation: Ridge is a strong linear baseline; tree models capture non-linear interactions.
                  The winner here is selected strictly by tuned RMSE.
                </p>
              </div>
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No tuning results loaded.
              </p>
            )}
          </>
        ) : null}

        {/* RAW JSON (lazy: only renders on this tab) */}
        {tab === "raw" ? (
          <>
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
                <h2 style={{ margin: "2px 0 6px", fontSize: 18, fontWeight: 900 }}>Raw JSON</h2>
                <p className="muted" style={{ fontSize: 13, margin: 0 }}>
                  For debugging / transparency. (This can be big.)
                </p>
              </div>
              <button className="btn" type="button" onClick={() => setTab("overview")}>
                Back to Overview
              </button>
            </div>

            {data ? (
              <pre
                style={{
                  marginTop: 12,
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
            ) : (
              <p className="muted" style={{ marginTop: 12 }}>
                No data loaded.
              </p>
            )}
          </>
        ) : null}
      </div>
    </section>
  );
}
