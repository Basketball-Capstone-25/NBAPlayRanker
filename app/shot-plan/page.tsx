// app/shot-plan/page.tsx

"use client";

import Link from "next/link";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchMetaOptions,
  fetchPbpMetaOptions,
  fetchShotHeatmap,
  fetchShotPlanRank,
  getShotPlanPdfUrl,
} from "../utils";

type ShotRow = Record<string, any>;

type Meta = {
  seasons: string[];
  teams: string[];
  shotTypes: string[];
  zones: string[];
  teamNames?: Record<string, string>;
};

type HeatmapPayload = {
  image_base64: string;
  caption?: string;
  n_shots?: number;
  n_shots_total?: number;
  n_shots_rendered?: number;
  _endpoint_used?: string;
  [key: string]: any;
};

type RankPayload = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  w_off: number;
  w_def: number;
  best_shooter?: Record<string, any>;
  top_shot_types: ShotRow[];
  top_zones: ShotRow[];
  top_pairs?: ShotRow[];
  notes?: string[];
  _endpoint_used?: string;
};

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

function fmtNum(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "-";
  return x.toFixed(digits);
}

function fmtInt(n: any) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "-";
  return Math.round(x).toLocaleString();
}

function getPrimaryScore(row: ShotRow) {
  const candidates = [
    row?.EPA_PRED,
    row?.PTS_PRED,
    row?.PPP_PRED,
    row?.pred,
    row?.score,
  ];
  for (const c of candidates) {
    const v = Number(c);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function getOffScore(row: ShotRow) {
  const candidates = [row?.EPA_OFF_SHRUNK, row?.PTS_OFF_SHRUNK, row?.PPP_OFF_SHRUNK];
  for (const c of candidates) {
    const v = Number(c);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function getDefScore(row: ShotRow) {
  const candidates = [row?.EPA_DEF_SHRUNK, row?.PTS_DEF_SHRUNK, row?.PPP_DEF_SHRUNK];
  for (const c of candidates) {
    const v = Number(c);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function getAttemptsOff(row: ShotRow) {
  const candidates = [row?.attempts_OFF, row?.ATTEMPTS_OFF, row?.att_off];
  for (const c of candidates) {
    const v = Number(c);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function getAttemptsDef(row: ShotRow) {
  const candidates = [row?.attempts_DEF, row?.ATTEMPTS_DEF, row?.att_def];
  for (const c of candidates) {
    const v = Number(c);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function getShotTypeLabel(row: ShotRow) {
  return String(row?.SHOT_TYPE ?? row?.shot_type ?? row?.type ?? "-");
}

function getZoneLabel(row: ShotRow) {
  return String(row?.ZONE ?? row?.zone ?? "-");
}

function pickDefaultSeason(seasons: string[]) {
  if (!seasons?.length) return "2025-26";
  return seasons[seasons.length - 1];
}

function pickDefaultTeam(teams: string[], prefer: string) {
  if (!teams?.length) return prefer;
  return teams.includes(prefer) ? prefer : teams[0];
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
          ? "linear-gradient(135deg, rgba(59,130,246,0.18), rgba(124,58,237,0.16))"
          : "rgba(255,255,255,0.045)",
        border: "1px solid rgba(255,255,255,0.10)",
      }}
    >
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.68)", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.35 }}>{value}</div>
    </div>
  );
}

function ActionChip({
  active,
  onClick,
  children,
}: {
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        borderRadius: 999,
        padding: "8px 13px",
        border: active ? "1px solid rgba(96,165,250,0.55)" : "1px solid rgba(255,255,255,0.12)",
        background: active
          ? "linear-gradient(135deg, rgba(37,99,235,0.24), rgba(124,58,237,0.20))"
          : "rgba(255,255,255,0.05)",
        color: "white",
        cursor: "pointer",
        fontSize: 13,
        fontWeight: 700,
        lineHeight: "16px",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </button>
  );
}

export default function ShotPlanPage() {
  const [meta, setMeta] = useState<Meta | null>(null);

  const [season, setSeason] = useState("");
  const [our, setOur] = useState("TOR");
  const [opp, setOpp] = useState("BOS");
  const [k, setK] = useState(5);
  const [wOff, setWOff] = useState(0.7);

  const [shotType, setShotType] = useState("");
  const [zone, setZone] = useState("");
  const [maxShots, setMaxShots] = useState(30000);

  const [autoRenderHeatmap, setAutoRenderHeatmap] = useState(true);
  const [showWhy, setShowWhy] = useState(true);

  const [loading, setLoading] = useState(false);
  const [loadingHeatmapOnly, setLoadingHeatmapOnly] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [rank, setRank] = useState<RankPayload | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapPayload | null>(null);

  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [mMain, mPbp] = await Promise.all([fetchMetaOptions(), fetchPbpMetaOptions()]);
        const merged: Meta = {
          seasons: mPbp?.seasons ?? [],
          teams: mPbp?.teams ?? [],
          shotTypes: mPbp?.shotTypes ?? [],
          zones: mPbp?.zones ?? [],
          teamNames: mMain?.teamNames ?? {},
        };

        if (cancelled) return;
        setMeta(merged);

        if (!didInitRef.current) {
          didInitRef.current = true;
          const saved = safeLocalGet("nbaPlayRanker_shotPlan_v2");
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              setSeason(String(parsed.season ?? ""));
              setOur(String(parsed.our ?? "TOR"));
              setOpp(String(parsed.opp ?? "BOS"));
              setK(clamp(Number(parsed.k ?? 5), 1, 10));
              setWOff(clamp(Number(parsed.wOff ?? 0.7), 0, 1));
              setShotType(String(parsed.shotType ?? ""));
              setZone(String(parsed.zone ?? ""));
              setMaxShots(clamp(Number(parsed.maxShots ?? 30000), 1000, 100000));
              setAutoRenderHeatmap(Boolean(parsed.autoRenderHeatmap ?? true));
              setShowWhy(Boolean(parsed.showWhy ?? true));
            } catch {
              // ignore
            }
          }
        }

        setSeason((prev) => prev || pickDefaultSeason(merged.seasons));
        setOur((prev) => prev || pickDefaultTeam(merged.teams, "TOR"));
        setOpp((prev) => prev || pickDefaultTeam(merged.teams, "BOS"));
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? "Failed to load Shot Plan options.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotPlan_v2",
      JSON.stringify({
        season,
        our,
        opp,
        k,
        wOff,
        shotType,
        zone,
        maxShots,
        autoRenderHeatmap,
        showWhy,
      })
    );
  }, [season, our, opp, k, wOff, shotType, zone, maxShots, autoRenderHeatmap, showWhy]);

  const wDef = useMemo(() => 1 - wOff, [wOff]);
  const canRun = Boolean(season && our && opp);
  const sameTeamSelected = Boolean(our && opp && our === opp);

  const teamLabel = (abbr: string) => {
    const name = meta?.teamNames?.[abbr];
    return name ? `${abbr} (${name})` : abbr;
  };

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
      maxShots,
    });
  }, [season, our, opp, k, wOff, shotType, zone, maxShots]);

  const topShotTypes: ShotRow[] = Array.isArray(rank?.top_shot_types) ? rank!.top_shot_types : [];
  const topZones: ShotRow[] = Array.isArray(rank?.top_zones) ? rank!.top_zones : [];
  const topPairs: ShotRow[] = Array.isArray(rank?.top_pairs) ? rank!.top_pairs : [];

  const bestShotType = topShotTypes[0] ?? null;
  const bestZone = topZones[0] ?? null;
  const bestPair = topPairs[0] ?? null;

  async function runFullPlan() {
    if (!canRun) {
      setError("Please select season, our team, and opponent.");
      return;
    }
    if (sameTeamSelected) {
      setError("Our team and opponent cannot be the same.");
      return;
    }

    const myId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    const rankPromise = fetchShotPlanRank({
      season,
      our,
      opp,
      k: clamp(k, 1, 10),
      wOff: clamp(wOff, 0, 1),
    });

    const heatmapPromise = fetchShotHeatmap({
      season,
      team: our,
      opp,
      shotType: shotType || undefined,
      zone: zone || undefined,
      maxShots,
    });

    const [rankRes, heatmapRes] = await Promise.allSettled([rankPromise, heatmapPromise]);

    if (requestIdRef.current !== myId) return;

    let nextError: string | null = null;

    if (rankRes.status === "fulfilled") {
      setRank(rankRes.value as RankPayload);
    } else {
      setRank(null);
      nextError = rankRes.reason?.message ?? "Shot Plan ranking failed.";
    }

    if (heatmapRes.status === "fulfilled") {
      setHeatmap(heatmapRes.value as HeatmapPayload);
    } else {
      setHeatmap(null);
      nextError = nextError
        ? `${nextError}\n\n${heatmapRes.reason?.message ?? "Heatmap rendering failed."}`
        : heatmapRes.reason?.message ?? "Heatmap rendering failed.";
    }

    setError(nextError);
    setLoading(false);
  }

  async function rerenderHeatmap(next?: { nextShotType?: string; nextZone?: string }) {
    if (!canRun || sameTeamSelected) return;

    const myId = ++requestIdRef.current;
    setLoadingHeatmapOnly(true);
    setError(null);

    try {
      const response = await fetchShotHeatmap({
        season,
        team: our,
        opp,
        shotType: (next?.nextShotType ?? shotType) || undefined,
        zone: (next?.nextZone ?? zone) || undefined,
        maxShots,
      });

      if (requestIdRef.current !== myId) return;
      setHeatmap(response as HeatmapPayload);
    } catch (e: any) {
      if (requestIdRef.current !== myId) return;
      setHeatmap(null);
      setError(e?.message ?? "Heatmap rendering failed.");
    } finally {
      if (requestIdRef.current === myId) setLoadingHeatmapOnly(false);
    }
  }

  function resetAll() {
    if (!meta) return;
    setSeason(pickDefaultSeason(meta.seasons));
    setOur(pickDefaultTeam(meta.teams, "TOR"));
    setOpp(pickDefaultTeam(meta.teams, "BOS"));
    setK(5);
    setWOff(0.7);
    setShotType("");
    setZone("");
    setMaxShots(30000);
    setError(null);
    setRank(null);
    setHeatmap(null);
  }

  function swapTeams() {
    setOur(opp);
    setOpp(our);
    setRank(null);
    setHeatmap(null);
  }

  function applyShotTypeToHeatmap(value: string) {
    setShotType(value);
    if (autoRenderHeatmap) {
      rerenderHeatmap({ nextShotType: value, nextZone: zone });
    }
  }

  function applyZoneToHeatmap(value: string) {
    setZone(value);
    if (autoRenderHeatmap) {
      rerenderHeatmap({ nextShotType: shotType, nextZone: value });
    }
  }

  const heroStyle: React.CSSProperties = {
    borderRadius: 24,
    padding: 20,
    background:
      "radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 32%), radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 26%), linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92))",
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
      <header className="page__header" style={heroStyle}>
        <div
          style={{
            display: "flex",
            gap: 16,
            alignItems: "flex-start",
            justifyContent: "space-between",
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
              Dataset2 • Baseline Shot Recommendation Engine
            </div>

            <h1 style={{ margin: 0, fontSize: "clamp(2rem, 4vw, 2.8rem)", lineHeight: 1.05 }}>
              Shot Plan
            </h1>

            <p className="muted" style={{ marginTop: 10, marginBottom: 0, maxWidth: 760 }}>
              Generate matchup-specific shot recommendations, rank the best shot types and zones,
              then connect the plan to a live heatmap so the visual stays aligned to the exact filters.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <Link className="btn btn--secondary" href="/shot-explorer">
              Open Shot Explorer
            </Link>
            <button className="btn" onClick={runFullPlan} disabled={loading || !canRun || sameTeamSelected}>
              {loading ? "Running Plan…" : "Run Shot Plan"}
            </button>
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
          <StatCard label="Matchup" value={`${our || "—"} vs ${opp || "—"}`} accent />
          <StatCard label="Season" value={season || "—"} />
          <StatCard label="Top K" value={k} />
          <StatCard label="Offense Weight" value={fmtNum(wOff, 2)} />
          <StatCard label="Defense Weight" value={fmtNum(wDef, 2)} />
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
            <h2 style={{ marginBottom: 6 }}>Filters</h2>
            <p className="muted" style={{ marginTop: 0, marginBottom: 0 }}>
              The ranking call uses season, matchup, Top K, and blend weights. The optional shot type
              and zone filters only narrow the heatmap visualization.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={showWhy}
                onChange={(e) => setShowWhy(e.target.checked)}
              />
              Show rationale
            </label>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={autoRenderHeatmap}
                onChange={(e) => setAutoRenderHeatmap(e.target.checked)}
              />
              Auto-render heatmap
            </label>
          </div>
        </div>

        <div className="form-grid" style={{ marginTop: 14 }}>
          <label>
            Season
            <select value={season} onChange={(e) => setSeason(e.target.value)} disabled={!meta?.seasons?.length}>
              {(meta?.seasons ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Our team
            <select value={our} onChange={(e) => setOur(e.target.value)} disabled={!meta?.teams?.length}>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select value={opp} onChange={(e) => setOpp(e.target.value)} disabled={!meta?.teams?.length}>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {teamLabel(t)}
                </option>
              ))}
            </select>
          </label>

          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="btn btn--secondary" type="button" onClick={swapTeams} disabled={!our || !opp}>
              Swap matchup ↔
            </button>
          </div>

          <label>
            Top K
            <input
              type="number"
              min={1}
              max={10}
              step={1}
              value={k}
              onChange={(e) => setK(clamp(Number(e.target.value || "1"), 1, 10))}
            />
            <div className="help">Backend validation requires 1–10.</div>
          </label>

          <label>
            Offense weight (wOff)
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={wOff}
              onChange={(e) => setWOff(clamp(Number(e.target.value || "0"), 0, 1))}
            />
            <div className="help">Higher favors your offense more heavily in the blend.</div>
          </label>

          <label>
            Heatmap max shots
            <input
              type="number"
              min={1000}
              max={100000}
              step={1000}
              value={maxShots}
              onChange={(e) => setMaxShots(clamp(Number(e.target.value || "0"), 1000, 100000))}
            />
            <div className="help">Higher keeps more shot events but can slow rendering.</div>
          </label>

          <label>
            Heatmap shot type (optional)
            <select value={shotType} onChange={(e) => setShotType(e.target.value)} disabled={!meta?.shotTypes?.length}>
              <option value="">All</option>
              {(meta?.shotTypes ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Heatmap zone (optional)
            <select value={zone} onChange={(e) => setZone(e.target.value)} disabled={!meta?.zones?.length}>
              <option value="">All</option>
              {(meta?.zones ?? []).map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div
          style={{
            marginTop: 14,
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <button className="btn" onClick={runFullPlan} disabled={loading || !canRun || sameTeamSelected}>
            {loading ? "Running Plan…" : "Run Plan"}
          </button>

          <button
            className="btn btn--secondary"
            type="button"
            onClick={() => rerenderHeatmap()}
            disabled={loadingHeatmapOnly || !rank || sameTeamSelected}
          >
            {loadingHeatmapOnly ? "Rendering Heatmap…" : "Refresh Heatmap Only"}
          </button>

          <button className="btn btn--secondary" type="button" onClick={resetAll} disabled={!meta}>
            Reset
          </button>

          {pdfUrl ? (
            <a className="btn btn--secondary" href={pdfUrl} target="_blank" rel="noreferrer">
              Export PDF
            </a>
          ) : null}

          <Link className="btn btn--secondary" href="/shot-heatmap">
            Open Full Heatmap Page
          </Link>
        </div>

        {sameTeamSelected ? (
          <p className="error" style={{ marginTop: 12 }}>
            Our team and opponent cannot be the same. Choose a real matchup.
          </p>
        ) : null}

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
      </section>

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
          <div>
            <h2 style={{ marginBottom: 6 }}>Plan Summary</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              {rank
                ? `${teamLabel(rank.our_team)} vs ${teamLabel(rank.opp_team)} • ${rank.season} • k=${rank.k}`
                : "Run the Shot Plan to generate recommendations."}
            </p>
          </div>
        </div>

        {!rank ? (
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
            No plan loaded yet. Click Run Plan to generate baseline shot recommendations.
          </div>
        ) : (
          <div className="viz" style={{ display: "grid", gap: 14, marginTop: 12 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 10,
              }}
            >
              <StatCard label="Best Shot Type" value={bestShotType ? getShotTypeLabel(bestShotType) : "-"} accent />
              <StatCard label="Best Zone" value={bestZone ? getZoneLabel(bestZone) : "-"} />
              <StatCard
                label="Best Shooter"
                value={rank.best_shooter?.PLAYER_NAME ?? rank.best_shooter?.player_name ?? "-"}
              />
              <StatCard label="Ranking Endpoint" value={rank._endpoint_used || "Backend API"} />
            </div>

            {rank.notes?.length ? (
              <div
                style={{
                  borderRadius: 16,
                  padding: "12px 14px",
                  background: "rgba(255,255,255,0.035)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.72)", marginBottom: 6 }}>
                  Notes
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.55 }}>
                  {rank.notes.map((n, i) => (
                    <li key={`${n}-${i}`}>{n}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </section>

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
          <div>
            <h2 style={{ marginBottom: 6 }}>Top Shot Types</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              Click any row action to push that recommendation directly into the heatmap filters.
            </p>
          </div>
        </div>

        {!topShotTypes.length ? (
          <p className="muted" style={{ marginTop: 12 }}>
            No shot-type recommendations loaded yet.
          </p>
        ) : (
          <div className="card" style={{ padding: 0, overflowX: "auto", marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>SHOT_TYPE</th>
                  <th>PRED</th>
                  <th>OFF</th>
                  <th>DEF</th>
                  <th>ATT_OFF</th>
                  <th>ATT_DEF</th>
                  <th>ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {topShotTypes.map((row, idx) => {
                  const shotTypeLabel = getShotTypeLabel(row);
                  const rationale = row?.RATIONALE ?? row?.rationale ?? "";
                  const active = shotType === shotTypeLabel;

                  return (
                    <React.Fragment key={`${shotTypeLabel}-${idx}`}>
                      <tr>
                        <td>{shotTypeLabel}</td>
                        <td>{fmtNum(getPrimaryScore(row))}</td>
                        <td>{fmtNum(getOffScore(row))}</td>
                        <td>{fmtNum(getDefScore(row))}</td>
                        <td>{fmtInt(getAttemptsOff(row))}</td>
                        <td>{fmtInt(getAttemptsDef(row))}</td>
                        <td>
                          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            <ActionChip active={active} onClick={() => applyShotTypeToHeatmap(shotTypeLabel)}>
                              Use in heatmap
                            </ActionChip>
                          </div>
                        </td>
                      </tr>

                      {showWhy && rationale ? (
                        <tr>
                          <td colSpan={7} style={{ background: "rgba(255,255,255,0.025)" }}>
                            <div style={{ padding: "6px 0", color: "rgba(255,255,255,0.78)" }}>
                              <strong>Why:</strong> {rationale}
                            </div>
                          </td>
                        </tr>
                      ) : null}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

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
          <div>
            <h2 style={{ marginBottom: 6 }}>Top Zones</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              Use zone recommendations to isolate the spatial profile of the matchup.
            </p>
          </div>
        </div>

        {!topZones.length ? (
          <p className="muted" style={{ marginTop: 12 }}>
            No zone recommendations loaded yet.
          </p>
        ) : (
          <div className="card" style={{ padding: 0, overflowX: "auto", marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>ZONE</th>
                  <th>PRED</th>
                  <th>OFF</th>
                  <th>DEF</th>
                  <th>ATT_OFF</th>
                  <th>ATT_DEF</th>
                  <th>ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {topZones.map((row, idx) => {
                  const zoneLabel = getZoneLabel(row);
                  const rationale = row?.RATIONALE ?? row?.rationale ?? "";
                  const active = zone === zoneLabel;

                  return (
                    <React.Fragment key={`${zoneLabel}-${idx}`}>
                      <tr>
                        <td>{zoneLabel}</td>
                        <td>{fmtNum(getPrimaryScore(row))}</td>
                        <td>{fmtNum(getOffScore(row))}</td>
                        <td>{fmtNum(getDefScore(row))}</td>
                        <td>{fmtInt(getAttemptsOff(row))}</td>
                        <td>{fmtInt(getAttemptsDef(row))}</td>
                        <td>
                          <ActionChip active={active} onClick={() => applyZoneToHeatmap(zoneLabel)}>
                            Use in heatmap
                          </ActionChip>
                        </td>
                      </tr>

                      {showWhy && rationale ? (
                        <tr>
                          <td colSpan={7} style={{ background: "rgba(255,255,255,0.025)" }}>
                            <div style={{ padding: "6px 0", color: "rgba(255,255,255,0.78)" }}>
                              <strong>Why:</strong> {rationale}
                            </div>
                          </td>
                        </tr>
                      ) : null}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

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
          <div>
            <h2 style={{ marginBottom: 6 }}>Top Shot Type + Zone Pairs</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              If your backend returns pair-level recommendations, they appear here.
            </p>
          </div>
        </div>

        {!topPairs.length ? (
          <p className="muted" style={{ marginTop: 12 }}>
            No pair-level recommendations returned by the backend.
          </p>
        ) : (
          <div className="card" style={{ padding: 0, overflowX: "auto", marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>SHOT_TYPE</th>
                  <th>ZONE</th>
                  <th>PRED</th>
                  <th>ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {topPairs.map((row, idx) => {
                  const shotTypeLabel = getShotTypeLabel(row);
                  const zoneLabel = getZoneLabel(row);

                  return (
                    <tr key={`${shotTypeLabel}-${zoneLabel}-${idx}`}>
                      <td>{shotTypeLabel}</td>
                      <td>{zoneLabel}</td>
                      <td>{fmtNum(getPrimaryScore(row))}</td>
                      <td>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                          <ActionChip
                            active={shotType === shotTypeLabel}
                            onClick={() => applyShotTypeToHeatmap(shotTypeLabel)}
                          >
                            Use shot type
                          </ActionChip>
                          <ActionChip
                            active={zone === zoneLabel}
                            onClick={() => applyZoneToHeatmap(zoneLabel)}
                          >
                            Use zone
                          </ActionChip>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {bestPair ? (
          <div
            style={{
              marginTop: 12,
              borderRadius: 16,
              padding: "12px 14px",
              background: "rgba(255,255,255,0.035)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div style={{ fontSize: 13, color: "rgba(255,255,255,0.72)", marginBottom: 6 }}>
              Best Pair Headline
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.55 }}>
              {getShotTypeLabel(bestPair)} from {getZoneLabel(bestPair)} currently grades best for this matchup.
            </div>
          </div>
        ) : null}
      </section>

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
          <div>
            <h2 style={{ marginBottom: 6 }}>Plan Heatmap</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              The heatmap below is tied to the current matchup plus the optional shot type and zone filters.
            </p>
          </div>

          {heatmap?.image_base64 ? (
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <button
                className="btn btn--secondary"
                type="button"
                onClick={() => rerenderHeatmap()}
                disabled={loadingHeatmapOnly}
              >
                {loadingHeatmapOnly ? "Rendering…" : "Re-render"}
              </button>
            </div>
          ) : null}
        </div>

        {!heatmap?.image_base64 ? (
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
            No heatmap loaded yet. Run the Shot Plan to render the visual.
          </div>
        ) : (
          <div className="viz" style={{ display: "grid", gap: 12, marginTop: 12 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 10,
              }}
            >
              <StatCard label="Current Shot Type Filter" value={shotType || "All"} accent />
              <StatCard label="Current Zone Filter" value={zone || "All"} />
              <StatCard label="Matched Shots" value={fmtInt(heatmap.n_shots_total ?? heatmap.n_shots)} />
              <StatCard label="Heatmap Endpoint" value={heatmap._endpoint_used || "Backend API"} />
            </div>

            <div
              style={{
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
                alignItems: "center",
              }}
            >
              {bestShotType ? (
                <ActionChip onClick={() => applyShotTypeToHeatmap(getShotTypeLabel(bestShotType))}>
                  Focus best shot type
                </ActionChip>
              ) : null}
              {bestZone ? (
                <ActionChip onClick={() => applyZoneToHeatmap(getZoneLabel(bestZone))}>
                  Focus best zone
                </ActionChip>
              ) : null}
              <ActionChip
                active={!shotType && !zone}
                onClick={() => {
                  setShotType("");
                  setZone("");
                  if (autoRenderHeatmap) {
                    rerenderHeatmap({ nextShotType: "", nextZone: "" });
                  }
                }}
              >
                Clear heatmap filters
              </ActionChip>
            </div>

            <div
              style={{
                borderRadius: 18,
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.12)",
                background:
                  "radial-gradient(circle at top, rgba(59,130,246,0.10), transparent 38%), rgba(255,255,255,0.025)",
                padding: 10,
              }}
            >
              <img
                src={`data:image/png;base64,${heatmap.image_base64}`}
                alt="Shot plan heatmap"
                style={{
                  width: "100%",
                  maxWidth: 1100,
                  height: "auto",
                  display: "block",
                  margin: "0 auto",
                  borderRadius: 14,
                }}
              />
            </div>

            <div
              style={{
                borderRadius: 16,
                padding: "12px 14px",
                background: "rgba(255,255,255,0.035)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div style={{ fontSize: 13, color: "rgba(255,255,255,0.72)", marginBottom: 6 }}>
                Analyst Note
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.55 }}>
                This visual should shift when you apply a recommended shot type or zone. That lets the plan
                page act as a real decision-support surface instead of a static table.
              </div>
            </div>

            {heatmap.caption ? (
              <div className="muted" style={{ fontSize: 13 }}>
                {heatmap.caption}
              </div>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}