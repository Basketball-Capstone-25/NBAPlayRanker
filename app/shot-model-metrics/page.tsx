// app/shot-model-metrics/page.tsx

"use client";

import Link from "next/link";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { fetchShotModelMetrics } from "../utils";

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

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
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

function StatCard({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      style={{
        borderRadius: 18,
        padding: "14px 16px",
        background: accent
          ? "linear-gradient(135deg, rgba(34,197,94,0.16), rgba(59,130,246,0.16))"
          : "rgba(255,255,255,0.045)",
        border: "1px solid rgba(255,255,255,0.10)",
      }}
    >
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.68)", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.35 }}>{value}</div>
    </div>
  );
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

  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;

    const raw = safeLocalGet("nbaPlayRanker_shotModelMetrics_v2");
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
      // ignore malformed localStorage
    }
  }, []);

  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotModelMetrics_v2",
      JSON.stringify({ nSplits, autoRefresh, refreshEverySec })
    );
  }, [nSplits, autoRefresh, refreshEverySec]);

  async function load({ silent = false, forceRefresh = false }: { silent?: boolean; forceRefresh?: boolean } = {}) {
    const myId = ++requestIdRef.current;

    try {
      setLoading(true);
      if (!silent) setStatusHint("Loading metrics…");
      setError(null);

      const res = await fetchShotModelMetrics(nSplits, forceRefresh);

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

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nSplits]);

  useEffect(() => {
    if (!autoRefresh) return;

    const t = window.setInterval(() => {
      load({ silent: true, forceRefresh: true });
    }, clamp(refreshEverySec, 15, 300) * 1000);

    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, refreshEverySec, nSplits]);

  const rows = useMemo(() => {
    return Array.isArray(data?.metrics) ? data!.metrics : [];
  }, [data]);

  const bestByRmse = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.RMSE_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => a.RMSE_mean - b.RMSE_mean)[0];
  }, [rows]);

  const bestByMae = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.MAE_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => a.MAE_mean - b.MAE_mean)[0];
  }, [rows]);

  const bestByR2 = useMemo(() => {
    if (!rows.length) return null;
    const valid = rows.filter((r) => Number.isFinite(Number(r.R2_mean)));
    if (!valid.length) return null;
    return [...valid].sort((a, b) => b.R2_mean - a.R2_mean)[0];
  }, [rows]);

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
      return { model: r.model, widthPct: clamp(score * 100, 0, 100) };
    });
  }, [rows]);

  const r2Bars = useMemo(() => {
    if (!rows.length) return [];
    const vals = rows.map((r) => Number(r.R2_mean)).filter((x) => Number.isFinite(x));
    if (!vals.length) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const denom = Math.max(1e-9, max - min);

    return rows.map((r) => {
      const v = Number(r.R2_mean);
      const score = Number.isFinite(v) ? (v - min) / denom : 0;
      return { model: r.model, widthPct: clamp(score * 100, 0, 100) };
    });
  }, [rows]);

  const canExport = rows.length > 0;

  const headerStyle: React.CSSProperties = {
    borderRadius: 24,
    padding: 20,
    background:
      "radial-gradient(circle at top left, rgba(34,197,94,0.18), transparent 32%), radial-gradient(circle at top right, rgba(59,130,246,0.16), transparent 28%), linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92))",
    border: "1px solid rgba(255,255,255,0.10)",
    boxShadow: "0 16px 50px rgba(2,6,23,0.35)",
    overflow: "hidden",
  };

  const panelStyle: React.CSSProperties = {
    borderRadius: 22,
    border: "1px solid rgba(255,255,255,0.10)",
    background: "linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02))",
    boxShadow: "0 12px 38px rgba(2,6,23,0.20)",
  };

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
          <div style={{ minWidth: 280, maxWidth: 820 }}>
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.10)",
                fontSize: 12,
                fontWeight: 700,
                letterSpacing: 0.2,
                marginBottom: 12,
              }}
            >
              Dataset2 • Holdout Evaluation
            </div>

            <h1 className="h1" style={{ margin: 0 }}>
              Shot Model Metrics
            </h1>

            <p className="muted" style={{ fontSize: 14, marginTop: 10, marginBottom: 0, maxWidth: 760 }}>
              Compare shot-level expected-points models on grouped holdout splits. Lower RMSE / MAE is
              better. Higher R² is better. GroupKFold by GAME_ID helps avoid game-level leakage.
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
            marginTop: 16,
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

            <button className="btn" type="button" onClick={() => load({ forceRefresh: true })} disabled={loading}>
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
            {lastUpdated ? (
              <span className="muted" style={{ fontSize: 12 }}>
                Updated: {new Date(lastUpdated).toLocaleTimeString()}
              </span>
            ) : null}
          </div>
        </div>

        <div
          style={{
            marginTop: 18,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 10,
          }}
        >
          <StatCard label="Fold Count" value={data?.n_splits ?? nSplits} accent />
          <StatCard label="Models Returned" value={rows.length} />
          <StatCard
            label="Best RMSE"
            value={bestByRmse ? `${bestByRmse.model} (${fmt(bestByRmse.RMSE_mean, 3)})` : "—"}
          />
          <StatCard
            label="Best MAE"
            value={bestByMae ? `${bestByMae.model} (${fmt(bestByMae.MAE_mean, 3)})` : "—"}
          />
          <StatCard
            label="Best R²"
            value={bestByR2 ? `${bestByR2.model} (${fmt(bestByR2.R2_mean, 3)})` : "—"}
          />
        </div>
      </header>

      <section className="card" style={{ ...panelStyle, marginTop: 14 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <h2 style={{ marginBottom: 6 }}>Performance Snapshot</h2>
            <p className="muted" style={{ marginTop: 0, marginBottom: 0 }}>
              Quick visual comparison so you can see the best generalization profile at a glance.
            </p>
          </div>

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
              disabled={!bestByRmse}
              onClick={async () => {
                if (!bestByRmse) return;
                const text = `Best shot model by RMSE: ${bestByRmse.model} | RMSE ${fmt(
                  bestByRmse.RMSE_mean,
                  3
                )} ± ${fmt(bestByRmse.RMSE_std, 3)} | MAE ${fmt(bestByRmse.MAE_mean, 3)} ± ${fmt(
                  bestByRmse.MAE_std,
                  3
                )} | R² ${fmt(bestByRmse.R2_mean, 3)} ± ${fmt(bestByRmse.R2_std, 3)}`;
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

        {error ? (
          <div
            style={{
              marginTop: 12,
              whiteSpace: "pre-wrap",
              borderRadius: 14,
              padding: "12px 14px",
              background: "rgba(239,68,68,0.10)",
              border: "1px solid rgba(239,68,68,0.28)",
              color: "rgb(254,202,202)",
            }}
          >
            {error}
          </div>
        ) : null}

        {rows.length === 0 && !loading ? (
          <div
            style={{
              borderRadius: 18,
              padding: "26px 18px",
              marginTop: 12,
              background: "rgba(255,255,255,0.03)",
              border: "1px dashed rgba(255,255,255,0.14)",
              textAlign: "center",
              color: "rgba(255,255,255,0.7)",
            }}
          >
            No shot model metrics returned yet. Click Refresh to load them.
          </div>
        ) : null}

        {rows.length > 0 ? (
          <div style={{ display: "grid", gap: 18, marginTop: 14 }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 10 }}>
                Lower RMSE is better
              </div>

              <div style={{ display: "grid", gap: 10 }}>
                {rmseBars.map((b) => {
                  const row = rows.find((r) => r.model === b.model);
                  const isBest = bestByRmse?.model === b.model;

                  return (
                    <div
                      key={`rmse-${b.model}`}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "220px 1fr 90px",
                        gap: 10,
                        alignItems: "center",
                      }}
                    >
                      <div style={{ fontSize: 13 }}>
                        <strong>{b.model}</strong>
                        <div className="muted" style={{ fontSize: 11 }}>
                          RMSE {fmt(row?.RMSE_mean, 3)} ± {fmt(row?.RMSE_std, 3)}
                        </div>
                      </div>

                      <div
                        style={{
                          height: 10,
                          borderRadius: 999,
                          background: "rgba(15,23,42,0.18)",
                          overflow: "hidden",
                          border: "1px solid rgba(255,255,255,0.10)",
                        }}
                      >
                        <div
                          style={{
                            width: `${b.widthPct}%`,
                            height: "100%",
                            background: isBest
                              ? "linear-gradient(90deg, rgba(34,197,94,0.70), rgba(59,130,246,0.55))"
                              : "rgba(148,163,184,0.55)",
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

            <div>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 10 }}>
                Higher R² is better
              </div>

              <div style={{ display: "grid", gap: 10 }}>
                {r2Bars.map((b) => {
                  const row = rows.find((r) => r.model === b.model);
                  const isBest = bestByR2?.model === b.model;

                  return (
                    <div
                      key={`r2-${b.model}`}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "220px 1fr 90px",
                        gap: 10,
                        alignItems: "center",
                      }}
                    >
                      <div style={{ fontSize: 13 }}>
                        <strong>{b.model}</strong>
                        <div className="muted" style={{ fontSize: 11 }}>
                          R² {fmt(row?.R2_mean, 3)} ± {fmt(row?.R2_std, 3)}
                        </div>
                      </div>

                      <div
                        style={{
                          height: 10,
                          borderRadius: 999,
                          background: "rgba(15,23,42,0.18)",
                          overflow: "hidden",
                          border: "1px solid rgba(255,255,255,0.10)",
                        }}
                      >
                        <div
                          style={{
                            width: `${b.widthPct}%`,
                            height: "100%",
                            background: isBest
                              ? "linear-gradient(90deg, rgba(96,165,250,0.75), rgba(168,85,247,0.55))"
                              : "rgba(148,163,184,0.55)",
                          }}
                        />
                      </div>

                      <div style={{ fontFamily: "var(--mono)", fontSize: 12, textAlign: "right" }}>
                        {fmt(row?.R2_mean, 3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : null}
      </section>

      {rows.length > 0 ? (
        <section className="card" style={panelStyle}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <h2 style={{ margin: "8px 0 6px", fontSize: 16 }}>Holdout Metrics Table</h2>
            <div className="muted" style={{ fontSize: 13 }}>
              Lower RMSE / MAE is better. Higher R² is better.
            </div>
          </div>

          <div style={{ overflowX: "auto", marginTop: 10 }}>
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
                  const isBestRmse = bestByRmse?.model === r.model;
                  const isBestMae = bestByMae?.model === r.model;
                  const isBestR2 = bestByR2?.model === r.model;

                  return (
                    <tr key={r.model}>
                      <td>
                        <strong>{r.model}</strong>
                        {isBestRmse ? (
                          <span className="badge blue" style={{ marginLeft: 8 }}>
                            best RMSE
                          </span>
                        ) : null}
                        {isBestMae ? (
                          <span className="badge" style={{ marginLeft: 8 }}>
                            best MAE
                          </span>
                        ) : null}
                        {isBestR2 ? (
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

      <section className="card" style={panelStyle}>
        <h2 style={{ marginTop: 0 }}>How to Read This</h2>
        <div style={{ display: "grid", gap: 10 }}>
          <div
            style={{
              borderRadius: 16,
              padding: "12px 14px",
              background: "rgba(255,255,255,0.035)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <strong>RMSE / MAE:</strong> average prediction error for shot-level expected points. Lower means
            the model stays closer to the actual result across held-out folds.
          </div>

          <div
            style={{
              borderRadius: 16,
              padding: "12px 14px",
              background: "rgba(255,255,255,0.035)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <strong>R²:</strong> how much of the variation in the target the model explains. Higher is better,
            but for noisy shot-level data it may still remain modest.
          </div>

          <div
            style={{
              borderRadius: 16,
              padding: "12px 14px",
              background: "rgba(255,255,255,0.035)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <strong>Grouped holdout:</strong> using GAME_ID as the grouping variable helps avoid leakage by
            keeping shots from the same game together inside either train or test.
          </div>
        </div>
      </section>
    </main>
  );
}