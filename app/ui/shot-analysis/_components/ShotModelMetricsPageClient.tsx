// Shot model metrics page UI.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchShotModelMetrics } from "../../../services/shotAnalysis";

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
};

function fmt(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

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
    // ignore
  }
}

function toCsv(rows: MetricsRow[]) {
  const header = ["model", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"];
  const lines = [header.join(",")];

  for (const r of rows) {
    const row = [
      r.model,
      String(r.RMSE_mean),
      String(r.RMSE_std),
      String(r.MAE_mean),
      String(r.MAE_std),
      String(r.R2_mean),
      String(r.R2_std),
    ];
    lines.push(row.map((x) => `"${String(x).replaceAll('"', '""')}"`).join(","));
  }

  return lines.join("\n");
}

export default function ShotModelMetricsPage() {
  const [nSplits, setNSplits] = useState<number>(5);
  const [data, setData] = useState<MetricsResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [statusHint, setStatusHint] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshEverySec, setRefreshEverySec] = useState<number>(60);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);

  // Restore prefs once (nSplits + refresh settings)
  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;

    const raw = safeLocalGet("nbaPlayRanker_shotModelMetrics_v1");
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw);
      const ns = Number(parsed.nSplits);
      const ar = Boolean(parsed.autoRefresh);
      const re = Number(parsed.refreshEverySec);

      if ([3, 4, 5, 6].includes(ns)) setNSplits(ns);
      if (Number.isFinite(re) && re >= 15 && re <= 300) setRefreshEverySec(re);
      setAutoRefresh(ar);
    } catch {
      // ignore
    }
  }, []);

  // Persist prefs
  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotModelMetrics_v1",
      JSON.stringify({ nSplits, autoRefresh, refreshEverySec })
    );
  }, [nSplits, autoRefresh, refreshEverySec]);

  async function load({ silent = false }: { silent?: boolean } = {}) {
    const myId = ++requestIdRef.current;

    try {
      setLoading(true);
      if (!silent) setStatusHint("Loading metrics…");
      setError(null);

      const res = await fetchShotModelMetrics(nSplits);

      // Ignore stale responses (prevents “wrong splits” UI after fast toggles).
      if (requestIdRef.current !== myId) return;

      setData(res);
      setLastUpdated(Date.now());
      setStatusHint("Updated ✅");
      window.setTimeout(() => setStatusHint(""), 900);
    } catch (e: any) {
      if (requestIdRef.current !== myId) return;

      console.error(e);
      setError(e?.message ?? "Failed to load shot model metrics.");
      setData(null);
      setStatusHint("Load failed");
      window.setTimeout(() => setStatusHint(""), 1200);
    } finally {
      if (requestIdRef.current === myId) setLoading(false);
    }
  }

  // Load on mount + whenever nSplits changes (single effect so we don't double-load)
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nSplits]);

  // Optional auto-refresh timer
  useEffect(() => {
    if (!autoRefresh) return;

    const t = window.setInterval(() => {
      // keep UI stable while refreshing
      load({ silent: true });
    }, Math.max(15, refreshEverySec) * 1000);

    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, refreshEverySec, nSplits]);

  const rows = useMemo(() => {
    return Array.isArray(data?.metrics) ? data!.metrics : [];
  }, [data]);

  // Best model is the one with lowest RMSE.
  const bestModel = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.RMSE_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => a.RMSE_mean - b.RMSE_mean)[0];
  }, [rows]);

  const bestR2 = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.R2_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => b.R2_mean - a.R2_mean)[0];
  }, [rows]);

  // Bar widths: normalize RMSE where lower RMSE is “better”, invert so better looks longer.
  const rmseBars = useMemo(() => {
    if (!rows.length) return [];
    const vals = rows.map((r) => Number(r.RMSE_mean)).filter((x) => Number.isFinite(x));
    if (!vals.length) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const denom = Math.max(1e-9, max - min);

    return rows.map((r) => {
      const v = Number(r.RMSE_mean);
      const score = Number.isFinite(v) ? 1 - (v - min) / denom : 0;
      return { model: r.model, widthPct: clamp01(score) * 100 };
    });
  }, [rows]);

  const headerStyle: React.CSSProperties = {
    borderRadius: 18,
    padding: "18px 18px 14px",
    background:
      "linear-gradient(135deg, rgba(34,197,94,0.12), rgba(99,102,241,0.16), rgba(56,189,248,0.14))",
    border: "1px solid rgba(255,255,255,0.10)",
  };

  const canExport = rows.length > 0;

  return (
    <main className="page" style={{ paddingBottom: 56 }}>
      <header className="page__header" style={headerStyle}>
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
              Shot Model Metrics
            </h1>
            <p className="muted" style={{ fontSize: 14, marginTop: 6, marginBottom: 0 }}>
              Holdout performance for shot-level expected points models (GroupKFold by GAME_ID).
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn btn--secondary" href="/shot-plan">
              Back: Shot Plan
            </Link>
            <Link className="btn" href="/shot-statistical-analysis">
              Next: Shot Statistical Analysis
            </Link>
          </div>
        </div>

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
            <label style={{ fontSize: 13 }}>
              n_splits (GroupKFold)
              <select
                className="input"
                style={{ width: 110, display: "inline-block", marginLeft: 8 }}
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

            <button className="btn" type="button" onClick={() => load()} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh"}
            </button>

            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>

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

            {statusHint ? <span className="badge">{statusHint}</span> : null}
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            {bestModel ? (
              <span className="badge blue">
                Best (lowest RMSE): {bestModel.model} ({fmt(bestModel.RMSE_mean, 3)})
              </span>
            ) : (
              <span className="badge">Best RMSE: —</span>
            )}

            {bestR2 ? (
              <span className="badge">
                Best R²: {bestR2.model} ({fmt(bestR2.R2_mean, 3)})
              </span>
            ) : null}

            {lastUpdated ? (
              <span className="muted" style={{ fontSize: 12 }}>
                Updated: {new Date(lastUpdated).toLocaleTimeString()}
              </span>
            ) : null}
          </div>
        </div>
      </header>

      <section className="card" style={{ marginTop: 14 }}>
        {error ? (
          <p className="muted" style={{ marginTop: 0 }}>
            {error}
          </p>
        ) : null}

        {rows.length === 0 && !loading ? (
          <div className="muted" style={{ padding: "6px 0" }}>
            No metrics returned yet. Try refresh.
          </div>
        ) : null}

        {rows.length > 0 ? (
          <div style={{ marginTop: 4 }}>
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
                Holdout comparison (lower RMSE is better)
              </h2>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button
                  className="btn btn--secondary"
                  type="button"
                  disabled={!canExport}
                  onClick={() => {
                    const csv = toCsv(rows);
                    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `shot_model_metrics_splits_${nSplits}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  Export CSV
                </button>

                <button
                  className="btn btn--secondary"
                  type="button"
                  disabled={!bestModel}
                  onClick={async () => {
                    if (!bestModel) return;
                    const text = `Best (lowest RMSE): ${bestModel.model} — RMSE ${fmt(
                      bestModel.RMSE_mean,
                      3
                    )} ± ${fmt(bestModel.RMSE_std, 3)} | MAE ${fmt(bestModel.MAE_mean, 3)} ± ${fmt(
                      bestModel.MAE_std,
                      3
                    )} | R² ${fmt(bestModel.R2_mean, 3)} ± ${fmt(bestModel.R2_std, 3)}`;
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
                  Copy best summary
                </button>
              </div>
            </div>

            <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
              {rmseBars.map((b) => {
                const row = rows.find((r) => r.model === b.model);
                const isBest = bestModel?.model === b.model;

                return (
                  <div
                    key={b.model}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "220px 1fr 90px",
                      gap: 10,
                      alignItems: "center",
                    }}
                  >
                    <div style={{ fontSize: 13 }}>
                      <strong title={`RMSE ${fmt(row?.RMSE_mean, 3)} ± ${fmt(row?.RMSE_std, 3)}`}>
                        {b.model}
                      </strong>
                      <div className="muted" style={{ fontSize: 11 }}>
                        RMSE {fmt(row?.RMSE_mean, 3)} ± {fmt(row?.RMSE_std, 3)}
                      </div>
                    </div>

                    <div
                      style={{
                        height: 10,
                        borderRadius: 999,
                        background: "rgba(15,23,42,0.08)",
                        overflow: "hidden",
                        border: "1px solid rgba(255,255,255,0.10)",
                      }}
                    >
                      <div
                        style={{
                          width: `${b.widthPct}%`,
                          height: "100%",
                          background: isBest
                            ? "linear-gradient(90deg, rgba(59,130,246,0.65), rgba(34,197,94,0.45))"
                            : "rgba(15, 23, 42, 0.30)",
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
          </div>
        ) : null}
      </section>

      {rows.length > 0 ? (
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
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>Holdout metrics table</h2>
            <div className="muted" style={{ fontSize: 13 }}>
              Tip: RMSE/MAE lower is better; R² higher is better.
            </div>
          </div>

          <div style={{ overflowX: "auto" }}>
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
                  const isBestR2 = bestR2?.model === r.model;

                  return (
                    <tr key={r.model}>
                      <td>
                        <strong>{r.model}</strong>
                        {isBest ? (
                          <span className="badge blue" style={{ marginLeft: 8 }}>
                            best RMSE
                          </span>
                        ) : null}
                        {isBestR2 && !isBest ? (
                          <span className="badge" style={{ marginLeft: 8 }}>
                            best R²
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
          </div>
        </section>
      ) : null}
    </main>
  );
}
