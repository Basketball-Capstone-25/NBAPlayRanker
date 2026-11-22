// Shot heatmap page UI.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import styles from "../../data-explorer/AnalystWorkspace.module.css";
import { useShotResult } from "./useShotResult";
import { loadShotOptions } from "./shotOptions";
import { ShotHeatmapCounts } from "./ShotHeatmapCounts";
import { shotHeatmapExportUrl } from "../../../domain/shot-heatmap-export";
import { fetchShotHeatmap } from "../../../services/shotAnalysis";

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
  const [team, setTeam] = useState<string>("");
  const [opp, setOpp] = useState<string>("");
  const [shotType, setShotType] = useState<string>("");
  const [zone, setZone] = useState<string>("");
  const [maxShots, setMaxShots] = useState<number>(35000);

  const [autoRender, setAutoRender] = useState<boolean>(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [metadataAttempt, setMetadataAttempt] = useState(0);
  const inputs = { season, team, opp, shotType, zone, maxShots };
  const query = useShotResult<typeof inputs, HeatmapResponse>(inputs);
  const { loading } = query;
  const error = metadataError ?? inputError ?? query.error;
  const heatmap = !error ? query.result : null;
  const didInitRef = useRef(false);

  // Load meta + restore saved settings (one-time restore guarded by didInitRef).
  useEffect(() => {
    let cancelled = false;
    setMetadataError(null);

    (async () => {
      try {
        const merged = await loadShotOptions();

        if (cancelled) return;

        setMeta(merged);

        // Restore saved inputs once so users can refresh and keep their last state.
        if (!didInitRef.current) {
          didInitRef.current = true;

          const saved = safeLocalGet("nbaPlayRanker_shotHeatmap_v1");
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              setSeason(merged.seasons.includes(parsed.season) ? parsed.season : "");
              setTeam(merged.teams.includes(parsed.team) ? parsed.team : "");
              setOpp(merged.teams.includes(parsed.opp) ? parsed.opp : "");
              setShotType(merged.shotTypes.includes(parsed.shotType) ? parsed.shotType : "");
              setZone(merged.zones.includes(parsed.zone) ? parsed.zone : "");
              const savedCap = Number(parsed.maxShots);
              setMaxShots(Number.isInteger(savedCap) && savedCap >= 1000 && savedCap <= 100000 ? savedCap : 35000);
              // SCRUM-843: auto-render always starts off, so nothing runs on page load.
            } catch {
              // ignore malformed stored data
            }
          }
        }

      } catch (e: any) {
        if (!cancelled) setMetadataError(e?.message ?? "Shot filters could not be loaded.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [metadataAttempt]);

  // Persist inputs so the page remembers your last “comparison setup.”
  useEffect(() => {
    // Initial defaults must not overwrite saved filters before metadata loads.
    if (!didInitRef.current) return;
    safeLocalSet(
      "nbaPlayRanker_shotHeatmap_v1",
      JSON.stringify({ season, team, opp, shotType, zone, maxShots, autoRender })
    );
  }, [season, team, opp, shotType, zone, maxShots, autoRender]);

  const subtitle = (() => {
    if (!query.inputs) return "";
    const { team, opp, season, shotType, zone, maxShots } = query.inputs;
    const parts = [
      `${normalizeLabel(team, meta?.teamNames)} vs ${normalizeLabel(opp, meta?.teamNames)}`,
      `Season: ${season || "—"}`,
    ];
    if (shotType) parts.push(`Shot Type: ${shotType}`);
    if (zone) parts.push(`Zone: ${zone}`);
    parts.push(`Max Shots: ${maxShots.toLocaleString()}`);
    return parts.join(" • ");
  })();

  // We require season + team + opponent so the backend has a defined matchup context.
  const canRun = Boolean(meta && season && team && opp && team !== opp);

  async function run() {
    setInputError(null);
    if (!canRun) {
      setInputError("Select a season and two different teams.");
      return;
    }
    if (!Number.isInteger(maxShots) || maxShots < 1000 || maxShots > 100000) {
      setInputError("Max shots must be a whole number from 1,000 to 100,000.");
      return;
    }
    await query.run(async ({ season, team, opp, shotType, zone, maxShots }) => {
      return await fetchShotHeatmap({
        season,
        team,
        opp,
        shotType: shotType || undefined,
        zone: zone || undefined,
        maxShots,
      });

    });
  }

  // Auto-render mode: rerender on any input change, but debounce to protect the backend.
  useEffect(() => {
    if (!autoRender) return;
    if (!canRun) return;

    const t = setTimeout(() => {
      run();
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
    // Reset clears every choice (SCRUM-841).
    if (!meta) return;
    setSeason("");
    setTeam("");
    setOpp("");
    setShotType("");
    setZone("");
    setMaxShots(35000);
    query.reset();
    setInputError(null);
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

  const downloadUrl = heatmap && query.inputs ? shotHeatmapExportUrl(query.inputs, heatmap.dataset_version) : null;

  return (
    <div className={`page ${styles.workspace}`} style={{ paddingBottom: 56 }}>
      <header className={styles.header}>
        <div className={styles.eyebrow}>Shot analysis</div>
        <div style={{ display: "flex", gap: 14, alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap" }}>
          <div style={{ minWidth: 0, flex: "1 1 300px" }}>
            <h1 style={{ margin: 0 }}>Shot Heatmap</h1>
            <p className="muted" style={{ marginTop: 6, marginBottom: 0 }}>
              Shot-attempt density against this opponent, including regular-season and playoff games.
              For ranked recommendations, use{" "}
              <Link href="/shot-plan" style={{ textDecoration: "underline" }}>
                Shot Plan
              </Link>
              .
            </p>
            <p className={styles.notice}>Color shows attempt counts, not shooting percentage. Makes and misses are both included.</p>
            <details><summary>How shots are shown</summary><p>Shots face the attacking basket. Attempts beyond the displayed half court are counted separately; color does not show expected points.</p></details>
          </div>

          <div className={styles.toolbar}>
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
            Choose a matchup, then render the court.
          </div>
        </div>

        <div className="form-grid">
          <label>
            Season
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)} disabled={!meta?.seasons?.length}>
              <option value="" disabled>Choose a season</option>
              {(meta?.seasons ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Team
            <select className="input" value={team} onChange={(e) => setTeam(e.target.value)} disabled={!meta?.teams?.length}>
              <option value="" disabled>Choose a team</option>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {normalizeLabel(t, meta?.teamNames)}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select className="input" value={opp} onChange={(e) => setOpp(e.target.value)} disabled={!meta?.teams?.length}>
              <option value="" disabled>Choose an opponent</option>
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
              className="input"
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
            <select className="input" value={zone} onChange={(e) => setZone(e.target.value)} disabled={!meta?.zones?.length}>
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
              className="input"
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
          <p role="alert" className="error" style={{ marginTop: 12 }}>
            {error}
              {metadataError && <button className="btn btn--secondary" type="button" onClick={() => setMetadataAttempt(value => value + 1)}>Retry filters</button>}
          </p>
        ) : null}
        {query.changed && <p role="status">Filters changed. Render again to see and download the updated heatmap.</p>}
      </section>

      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginBottom: 6 }}>Heatmap</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              {subtitle || "Render a heatmap to see the selected matchup."}
            </p>
          </div>

          {heatmap?.image_base64 ? (
            <div className={styles.toolbar}>
              {downloadUrl ? <a
                className="btn btn--secondary"
                href={downloadUrl}
                target="_blank"
                rel="noopener noreferrer"
                title="Downloads the displayed filters and data version. Any download error opens in a new tab."
              >
                Download PNG
              </a> : <span role="status">Render again to enable PNG download.</span>}
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
                title="Copies the filters used to render this image"
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

            <ShotHeatmapCounts heatmap={heatmap} />
          </div>
        )}
      </section>
    </div>
  );
}
