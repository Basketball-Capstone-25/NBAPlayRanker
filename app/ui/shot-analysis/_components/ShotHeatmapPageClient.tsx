// Shot heatmap page UI.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchMetaOptions, fetchPbpMetaOptions, fetchShotHeatmap } from "../../../services/shotAnalysis";

type Meta = {
  seasons: string[];
  teams: string[];
  shotTypes: string[];
  zones: string[];
  teamNames?: Record<string, string>;
};

type HeatmapResponse = {
  image_base64: string;
  caption?: string;
  n_shots?: number;
  // keep open-ended so we don't TS-break if backend adds fields
  [k: string]: any;
};

function pickDefaultSeason(seasons: string[]) {
  // Default to most recent season (or a stable fallback if meta is missing).
  if (!seasons?.length) return "2021-22";
  return seasons[seasons.length - 1];
}

function pickDefaultTeam(teams: string[], prefer: string) {
  // Prefer a common team so first load feels “ready” (TOR/BOS), otherwise use first option.
  if (!teams?.length) return prefer;
  return teams.includes(prefer) ? prefer : teams[0];
}

function clamp(n: number, lo: number, hi: number) {
  // Keeps numeric inputs within safe bounds so we don’t request absurd backend jobs.
  return Math.max(lo, Math.min(hi, n));
}

function safeLocalGet(key: string) {
  // localStorage can throw in some environments (Safari privacy / disabled storage).
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeLocalSet(key: string, value: string) {
  // Same reason as safeLocalGet: never let storage errors break the page.
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

function normalizeLabel(abbr: string, teamNames?: Record<string, string>) {
  // Converts "TOR" -> "TOR — Toronto Raptors" when name mapping exists.
  const name = teamNames?.[abbr];
  return name ? `${abbr} — ${name}` : abbr;
}

function findBestOption(options: string[], needles: string[]) {
  // Best-effort preset resolver:
  // 1) exact match (case-insensitive)
  // 2) substring match (case-insensitive)
  const opts = (options ?? []).filter(Boolean);
  if (!opts.length) return "";
  const lower = opts.map((x) => x.toLowerCase());

  for (const n of needles) {
    const needle = n.toLowerCase();
    const idx = lower.findIndex((o) => o === needle);
    if (idx >= 0) return opts[idx];
  }
  for (const n of needles) {
    const needle = n.toLowerCase();
    const idx = lower.findIndex((o) => o.includes(needle));
    if (idx >= 0) return opts[idx];
  }
  return "";
}

export default function ShotHeatmapPage() {
  const [meta, setMeta] = useState<Meta | null>(null);

  // Core filters (these map directly to the heatmap endpoint params).
  const [season, setSeason] = useState<string>("");
  const [team, setTeam] = useState<string>("TOR");
  const [opp, setOpp] = useState<string>("BOS");
  const [shotType, setShotType] = useState<string>("");
  const [zone, setZone] = useState<string>("");
  const [maxShots, setMaxShots] = useState<number>(35000);

  // Auto-render is for exploration mode; manual render is for “final screenshot” mode.
  const [autoRender, setAutoRender] = useState<boolean>(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapResponse | null>(null);

  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);

  // Load meta + restore saved settings (one-time restore guarded by didInitRef).
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [mMain, mPbp] = await Promise.all([fetchMetaOptions(), fetchPbpMetaOptions()]);

        const merged: Meta = {
          seasons: (mPbp?.seasons ?? []) as string[],
          teams: (mPbp?.teams ?? []) as string[],
          shotTypes: (mPbp?.shotTypes ?? []) as string[],
          zones: (mPbp?.zones ?? []) as string[],
          teamNames: (mMain?.teamNames ?? {}) as Record<string, string>,
        };

        if (cancelled) return;

        setMeta(merged);

        // Restore saved inputs once so users can refresh and keep their last state.
        if (!didInitRef.current) {
          didInitRef.current = true;

          const saved = safeLocalGet("nbaPlayRanker_shotHeatmap_v1");
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              setSeason(String(parsed.season ?? ""));
              setTeam(String(parsed.team ?? "TOR"));
              setOpp(String(parsed.opp ?? "BOS"));
              setShotType(String(parsed.shotType ?? ""));
              setZone(String(parsed.zone ?? ""));
              setMaxShots(Number(parsed.maxShots ?? 35000));
              setAutoRender(Boolean(parsed.autoRender ?? false));
            } catch {
              // ignore malformed stored data
            }
          }
        }

        // Defaults if empty (these only fill blanks; they don't overwrite restored values).
        setSeason((prev) => prev || pickDefaultSeason(merged.seasons));
        setTeam((prev) => prev || pickDefaultTeam(merged.teams, "TOR"));
        setOpp((prev) => prev || pickDefaultTeam(merged.teams, "BOS"));
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? "Failed to load meta options.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Persist inputs so the page remembers your last “comparison setup.”
  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotHeatmap_v1",
      JSON.stringify({ season, team, opp, shotType, zone, maxShots, autoRender })
    );
  }, [season, team, opp, shotType, zone, maxShots, autoRender]);

  const subtitle = useMemo(() => {
    // Human-readable “receipt” of the heatmap context (useful for copy/paste or screenshots).
    const parts = [
      `${normalizeLabel(team, meta?.teamNames)} vs ${normalizeLabel(opp, meta?.teamNames)}`,
      `Season: ${season || "—"}`,
    ];
    if (shotType) parts.push(`Shot Type: ${shotType}`);
    if (zone) parts.push(`Zone: ${zone}`);
    parts.push(`Max Shots: ${maxShots.toLocaleString()}`);
    return parts.join(" • ");
  }, [team, opp, season, shotType, zone, maxShots, meta?.teamNames]);

  // We require season + team + opponent so the backend has a defined matchup context.
  const canRun = Boolean(season && team && opp);

  async function run({ silent = false }: { silent?: boolean } = {}) {
    // silent=true means “keep the previous image visible while fetching a refresh.”
    if (!canRun) {
      if (!silent) setError("Please select season, team, and opponent.");
      return;
    }

    const myId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    try {
      // In normal runs, clear prior image so the user sees a fresh render state.
      if (!silent) setHeatmap(null);

      const res = await fetchShotHeatmap({
        season,
        team,
        opp,
        shotType: shotType || undefined,
        zone: zone || undefined,
        maxShots,
      });

      // Ignore stale responses (prevents “old render overwrote new render” bugs).
      if (requestIdRef.current !== myId) return;

      setHeatmap(res as HeatmapResponse);
    } catch (e: any) {
      if (requestIdRef.current !== myId) return;
      setHeatmap(null);
      setError(e?.message ?? "Failed to render heatmap.");
    } finally {
      if (requestIdRef.current === myId) setLoading(false);
    }
  }

  // Auto-render mode: rerender on any input change, but debounce to protect the backend.
  useEffect(() => {
    if (!autoRender) return;
    if (!canRun) return;

    const t = setTimeout(() => {
      // silent mode keeps the previous image displayed while refreshing
      run({ silent: true });
    }, 350);

    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRender, season, team, opp, shotType, zone, maxShots]);

  function swapTeams() {
    // Simple matchup flip for quick A/B comparisons.
    setTeam(opp);
    setOpp(team);
  }

  function resetFilters() {
    // Reset returns the page to a stable “default matchup” state.
    if (!meta) return;
    setSeason(pickDefaultSeason(meta.seasons));
    setTeam(pickDefaultTeam(meta.teams, "TOR"));
    setOpp(pickDefaultTeam(meta.teams, "BOS"));
    setShotType("");
    setZone("");
    setMaxShots(35000);
    setHeatmap(null);
    setError(null);
  }

  function applyPreset(preset: "ALL" | "RIM" | "CORNER_3" | "ABOVE_BREAK_3" | "MIDRANGE") {
    // Presets are best-effort mappings because dataset naming can vary.
    const shotTypes = meta?.shotTypes ?? [];
    const zones = meta?.zones ?? [];

    // Always start clean
    let nextShotType = "";
    let nextZone = "";

    if (preset === "ALL") {
      nextShotType = "";
      nextZone = "";
    }

    if (preset === "RIM") {
      nextShotType =
        findBestOption(shotTypes, ["rim", "at rim", "restricted", "paint"]) || nextShotType;
      nextZone = findBestOption(zones, ["rim", "restricted", "paint", "at rim"]) || nextZone;
    }

    if (preset === "CORNER_3") {
      nextShotType =
        findBestOption(shotTypes, ["corner 3", "corner three", "3pt", "3"]) || nextShotType;
      nextZone = findBestOption(zones, ["corner", "corner 3", "corner three"]) || nextZone;
    }

    if (preset === "ABOVE_BREAK_3") {
      nextShotType = findBestOption(shotTypes, ["3pt", "3", "three"]) || nextShotType;
      nextZone =
        findBestOption(zones, ["above the break", "above break", "arc", "3"]) || nextZone;
    }

    if (preset === "MIDRANGE") {
      nextShotType = findBestOption(shotTypes, ["mid", "midrange", "2pt"]) || nextShotType;
      nextZone = findBestOption(zones, ["mid", "midrange", "elbow", "short"]) || nextZone;
    }

    setShotType(nextShotType);
    setZone(nextZone);
  }

  const headerStyle: React.CSSProperties = {
    // Hero styling: the goal is to make this look like a finished “product” page.
    borderRadius: 18,
    padding: "18px 18px 14px",
    background:
      "linear-gradient(135deg, rgba(56,189,248,0.18), rgba(99,102,241,0.16), rgba(244,63,94,0.10))",
    border: "1px solid rgba(255,255,255,0.10)",
  };

  const chipStyle = (active: boolean): React.CSSProperties => ({
    // Preset chips: lightweight, no external UI libs.
    borderRadius: 999,
    padding: "8px 12px",
    border: "1px solid rgba(255,255,255,0.16)",
    background: active ? "rgba(99,102,241,0.18)" : "rgba(255,255,255,0.06)",
    cursor: "pointer",
    fontSize: 13,
    lineHeight: "16px",
    userSelect: "none",
    whiteSpace: "nowrap",
  });

  const presets = useMemo(
    () => [
      { key: "ALL" as const, label: "All shots" },
      { key: "RIM" as const, label: "Rim pressure" },
      { key: "CORNER_3" as const, label: "Corner 3s" },
      { key: "ABOVE_BREAK_3" as const, label: "Above-break 3s" },
      { key: "MIDRANGE" as const, label: "Midrange" },
    ],
    []
  );

  const downloadName = useMemo(() => {
    // Generates a stable, filesystem-safe filename for the PNG download.
    const safe = (s: string) => s.replace(/[^a-z0-9_\-]+/gi, "_");
    const pieces = [
      "shot_heatmap",
      season || "season",
      team || "team",
      "vs",
      opp || "opp",
      shotType ? safe(shotType) : "alltypes",
      zone ? safe(zone) : "allzones",
    ];
    return pieces.join("_") + ".png";
  }, [season, team, opp, shotType, zone]);

  return (
    <main className="page" style={{ paddingBottom: 56 }}>
      <header className="page__header" style={headerStyle}>
        <div style={{ display: "flex", gap: 14, alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap" }}>
          <div style={{ minWidth: 260 }}>
            <h1 style={{ margin: 0 }}>Shot Heatmap</h1>
            <p className="muted" style={{ marginTop: 6, marginBottom: 0 }}>
              Real, on-court shot density from Dataset2 (play-by-play shots).
              For ranked recommendations, use{" "}
              <Link href="/shot-plan" style={{ textDecoration: "underline" }}>
                Shot Plan
              </Link>
              .
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <Link className="btn btn--secondary" href="/shot-plan">
              Back to Shot Plan
            </Link>
            <button className="btn" onClick={() => run()} disabled={loading || !canRun}>
              {loading ? "Rendering…" : "Render"}
            </button>
          </div>
        </div>

        <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <div className="muted" style={{ fontSize: 13 }}>
            Quick presets:
          </div>
          {presets.map((p) => (
            <button
              key={p.key}
              type="button"
              style={chipStyle(false)}
              onClick={() => applyPreset(p.key)}
              title="Sets shot type + zone (when available in your metadata)"
            >
              {p.label}
            </button>
          ))}

          <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
              <input
                type="checkbox"
                checked={autoRender}
                onChange={(e) => setAutoRender(e.target.checked)}
              />
              Auto-render on change
            </label>
            <button className="btn btn--secondary" type="button" onClick={resetFilters} disabled={!meta}>
              Reset
            </button>
          </div>
        </div>
      </header>

      <section className="card" style={{ marginTop: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h2 style={{ marginBottom: 8 }}>Filters</h2>
          <div className="muted" style={{ fontSize: 13 }}>
            Tip: Keep Auto-render on during exploration, turn it off if your backend is busy.
          </div>
        </div>

        <div className="form-grid">
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
            Team
            <select value={team} onChange={(e) => setTeam(e.target.value)} disabled={!meta?.teams?.length}>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {normalizeLabel(t, meta?.teamNames)}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select value={opp} onChange={(e) => setOpp(e.target.value)} disabled={!meta?.teams?.length}>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {normalizeLabel(t, meta?.teamNames)}
                </option>
              ))}
            </select>
          </label>

          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="btn btn--secondary" type="button" onClick={swapTeams} disabled={!team || !opp}>
              Swap teams ↔
            </button>
          </div>

          <label>
            Shot type (optional)
            <select
              value={shotType}
              onChange={(e) => setShotType(e.target.value)}
              disabled={!meta?.shotTypes?.length}
            >
              <option value="">All</option>
              {(meta?.shotTypes ?? []).map((st) => (
                <option key={st} value={st}>
                  {st}
                </option>
              ))}
            </select>
          </label>

          <label>
            Zone (optional)
            <select value={zone} onChange={(e) => setZone(e.target.value)} disabled={!meta?.zones?.length}>
              <option value="">All</option>
              {(meta?.zones ?? []).map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </label>

          <label>
            Max shots (downsample)
            <input
              type="number"
              min={1000}
              max={100000}
              step={1000}
              value={maxShots}
              onChange={(e) => setMaxShots(clamp(Number(e.target.value || "0"), 1000, 100000))}
            />
            <div className="help">
              If there are more shots than this, the backend samples for speed.
            </div>
          </label>
        </div>

        {error ? (
          <p className="error" style={{ marginTop: 12 }}>
            {error}
          </p>
        ) : null}
      </section>

      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginBottom: 6 }}>Heatmap</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              {subtitle}
            </p>
          </div>

          {heatmap?.image_base64 ? (
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <a
                className="btn btn--secondary"
                href={`data:image/png;base64,${heatmap.image_base64}`}
                download={downloadName}
              >
                Download PNG
              </a>
              <button
                className="btn btn--secondary"
                type="button"
                onClick={async () => {
                  // Copies filter context so you can paste it into notes/defense slides.
                  try {
                    await navigator.clipboard.writeText(subtitle);
                  } catch {
                    // ignore
                  }
                }}
                title="Copies the current filter summary"
              >
                Copy summary
              </button>
            </div>
          ) : null}
        </div>

        {!heatmap?.image_base64 ? (
          <div className="muted" style={{ padding: "14px 0" }}>
            {loading ? "Rendering…" : "Click “Render” to display an image."}
          </div>
        ) : (
          <div className="viz" style={{ display: "grid", gap: 10 }}>
            <div
              style={{
                borderRadius: 14,
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.03)",
              }}
            >
              {/* We use <img> because the backend already returns a base64 PNG string. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${heatmap.image_base64}`}
                alt="Shot heatmap"
                style={{ width: "100%", maxWidth: 980, height: "auto", display: "block", margin: "0 auto" }}
              />
            </div>

            {heatmap.caption ? (
              <div className="muted" style={{ fontSize: 13 }}>
                {heatmap.caption}
              </div>
            ) : null}

            {typeof heatmap.n_shots === "number" ? (
              <div className="muted" style={{ fontSize: 13 }}>
                Shots used: {heatmap.n_shots.toLocaleString()}
              </div>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}
