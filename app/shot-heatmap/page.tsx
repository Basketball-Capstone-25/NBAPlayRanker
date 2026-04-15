// app/shot-heatmap/page.tsx

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { fetchMetaOptions, fetchPbpMetaOptions, fetchShotHeatmap } from "../utils";

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
  _endpoint_used?: string;
  [k: string]: any;
};

type PresetKey = "ALL" | "RIM" | "CORNER_3" | "ABOVE_BREAK_3" | "MIDRANGE";

function pickDefaultSeason(seasons: string[]) {
  if (!seasons?.length) return "2025-26";
  return seasons[seasons.length - 1];
}

function pickDefaultTeam(teams: string[], prefer: string) {
  if (!teams?.length) return prefer;
  return teams.includes(prefer) ? prefer : teams[0];
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
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

function normalizeLabel(abbr: string, teamNames?: Record<string, string>) {
  const name = teamNames?.[abbr];
  return name ? `${abbr} (${name})` : abbr;
}

function findBestOption(options: string[], needles: string[]) {
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
          ? "linear-gradient(135deg, rgba(59,130,246,0.18), rgba(124,58,237,0.18))"
          : "rgba(255,255,255,0.045)",
        border: "1px solid rgba(255,255,255,0.10)",
        boxShadow: accent ? "0 10px 30px rgba(59,130,246,0.10)" : "none",
      }}
    >
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.68)", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.35 }}>{value}</div>
    </div>
  );
}

export default function ShotHeatmapPage() {
  const [meta, setMeta] = useState<Meta | null>(null);

  const [season, setSeason] = useState<string>("");
  const [team, setTeam] = useState<string>("TOR");
  const [opp, setOpp] = useState<string>("BOS");
  const [shotType, setShotType] = useState<string>("");
  const [zone, setZone] = useState<string>("");
  const [maxShots, setMaxShots] = useState<number>(30000);

  const [autoRender, setAutoRender] = useState<boolean>(true);
  const [selectedPreset, setSelectedPreset] = useState<PresetKey>("ALL");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapResponse | null>(null);

  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);
  const didInitialRenderRef = useRef(false);

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

        if (!didInitRef.current) {
          didInitRef.current = true;
          const saved = safeLocalGet("nbaPlayRanker_shotHeatmap_v2");
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              setSeason(String(parsed.season ?? ""));
              setTeam(String(parsed.team ?? "TOR"));
              setOpp(String(parsed.opp ?? "BOS"));
              setShotType(String(parsed.shotType ?? ""));
              setZone(String(parsed.zone ?? ""));
              setMaxShots(Number(parsed.maxShots ?? 30000));
              setAutoRender(Boolean(parsed.autoRender ?? true));
              setSelectedPreset((parsed.selectedPreset as PresetKey) ?? "ALL");
            } catch {
              // ignore malformed local storage
            }
          }
        }

        setSeason((prev) => prev || pickDefaultSeason(merged.seasons));
        setTeam((prev) => prev || pickDefaultTeam(merged.teams, "TOR"));
        setOpp((prev) => prev || pickDefaultTeam(merged.teams, "BOS"));
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? "Failed to load heatmap options.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_shotHeatmap_v2",
      JSON.stringify({
        season,
        team,
        opp,
        shotType,
        zone,
        maxShots,
        autoRender,
        selectedPreset,
      })
    );
  }, [season, team, opp, shotType, zone, maxShots, autoRender, selectedPreset]);

  const canRun = Boolean(season && team && opp);
  const sameTeamSelected = Boolean(team && opp && team === opp);

  const subtitle = useMemo(() => {
    const parts = [
      `${normalizeLabel(team, meta?.teamNames)} vs ${normalizeLabel(opp, meta?.teamNames)}`,
      `Season: ${season || "—"}`,
      `Type: ${shotType || "All"}`,
      `Zone: ${zone || "All"}`,
      `Max shots: ${maxShots.toLocaleString()}`,
    ];
    return parts.join(" • ");
  }, [team, opp, season, shotType, zone, maxShots, meta?.teamNames]);

  const downloadName = useMemo(() => {
    const safe = (s: string) => s.replace(/[^a-z0-9_\-]+/gi, "_");
    return [
      "shot_heatmap",
      safe(season || "season"),
      safe(team || "team"),
      "vs",
      safe(opp || "opp"),
      safe(shotType || "all_types"),
      safe(zone || "all_zones"),
    ].join("_") + ".png";
  }, [season, team, opp, shotType, zone]);

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

  async function run({ silent = false }: { silent?: boolean } = {}) {
    if (!canRun) {
      if (!silent) setError("Please select season, team, and opponent.");
      return;
    }

    if (sameTeamSelected) {
      setError("Team and opponent cannot be the same. Choose a real matchup.");
      setHeatmap(null);
      return;
    }

    const myId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    try {
      if (!silent) setHeatmap(null);

      const res = await fetchShotHeatmap({
        season,
        team,
        opp,
        shotType: shotType || undefined,
        zone: zone || undefined,
        maxShots,
      });

      if (requestIdRef.current !== myId) return;
      setHeatmap(res as HeatmapResponse);
    } catch (e: any) {
      if (requestIdRef.current !== myId) return;
      setHeatmap(null);
      setError(e?.message ?? "Failed to render shot heatmap.");
    } finally {
      if (requestIdRef.current === myId) setLoading(false);
    }
  }

  useEffect(() => {
    if (!autoRender || !canRun || sameTeamSelected) return;

    const t = setTimeout(() => {
      run({ silent: true });
    }, 350);

    return () => clearTimeout(t);
  }, [autoRender, season, team, opp, shotType, zone, maxShots]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!meta || !canRun || sameTeamSelected || didInitialRenderRef.current) return;
    didInitialRenderRef.current = true;
    run();
  }, [meta, canRun, sameTeamSelected]); // eslint-disable-line react-hooks/exhaustive-deps

  function swapTeams() {
    setTeam(opp);
    setOpp(team);
  }

  function resetFilters() {
    if (!meta) return;
    setSeason(pickDefaultSeason(meta.seasons));
    setTeam(pickDefaultTeam(meta.teams, "TOR"));
    setOpp(pickDefaultTeam(meta.teams, "BOS"));
    setShotType("");
    setZone("");
    setMaxShots(30000);
    setSelectedPreset("ALL");
    setError(null);
    setHeatmap(null);
  }

  function applyPreset(preset: PresetKey) {
    const shotTypes = meta?.shotTypes ?? [];
    const zones = meta?.zones ?? [];

    let nextShotType = "";
    let nextZone = "";

    if (preset === "RIM") {
      nextShotType = findBestOption(shotTypes, ["layup", "dunk", "rim", "paint", "restricted"]);
      nextZone = findBestOption(zones, ["rim", "restricted", "paint", "at rim"]);
    } else if (preset === "CORNER_3") {
      nextShotType = findBestOption(shotTypes, ["jump shot", "3pt", "three", "pullup"]);
      nextZone = findBestOption(zones, ["corner3", "corner 3", "corner"]);
    } else if (preset === "ABOVE_BREAK_3") {
      nextShotType = findBestOption(shotTypes, ["jump shot", "3pt", "three", "pullup"]);
      nextZone = findBestOption(zones, ["arc3", "above break", "arc", "three"]);
    } else if (preset === "MIDRANGE") {
      nextShotType = findBestOption(shotTypes, ["jump shot", "fadeaway", "hook", "pullup", "bank shot", "mid"]);
      nextZone = findBestOption(zones, ["mid", "midrange"]);
    }

    setSelectedPreset(preset);
    setShotType(nextShotType);
    setZone(nextZone);
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

  const chipStyle = (active: boolean): React.CSSProperties => ({
    borderRadius: 999,
    padding: "9px 14px",
    border: active ? "1px solid rgba(96,165,250,0.55)" : "1px solid rgba(255,255,255,0.12)",
    background: active
      ? "linear-gradient(135deg, rgba(37,99,235,0.24), rgba(124,58,237,0.20))"
      : "rgba(255,255,255,0.05)",
    color: "white",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 700,
    lineHeight: "16px",
    userSelect: "none",
    whiteSpace: "nowrap",
    transition: "all 120ms ease",
  });

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
              Dataset2 • Dynamic Shot Density
            </div>

            <h1 style={{ margin: 0, fontSize: "clamp(2rem, 4vw, 2.8rem)", lineHeight: 1.05 }}>
              Shot Heatmap
            </h1>

            <p className="muted" style={{ marginTop: 10, marginBottom: 0, maxWidth: 760 }}>
              Render matchup-specific court heatmaps directly from the filtered play-by-play shot data.
              Change season, matchup, shot type, or zone and the visualization updates to reflect only
              that slice of the dataset.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <Link className="btn btn--secondary" href="/shot-plan">
              Go to Shot Plan
            </Link>
            <button className="btn" onClick={() => run()} disabled={loading || !canRun || sameTeamSelected}>
              {loading ? "Rendering…" : "Render Heatmap"}
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
          <StatCard label="Matchup" value={`${team || "—"} vs ${opp || "—"}`} accent />
          <StatCard label="Season" value={season || "—"} />
          <StatCard label="Shot Type Filter" value={shotType || "All"} />
          <StatCard label="Zone Filter" value={zone || "All"} />
          <StatCard label="Sampling Cap" value={maxShots.toLocaleString()} />
        </div>

        <div
          style={{
            marginTop: 16,
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <span className="muted" style={{ fontSize: 13 }}>
            Quick presets:
          </span>
          {presets.map((p) => (
            <button
              key={p.key}
              type="button"
              style={chipStyle(selectedPreset === p.key)}
              onClick={() => applyPreset(p.key)}
            >
              {p.label}
            </button>
          ))}

          <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
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
              Use exact matchup filters so every render reflects only the relevant shot subset.
            </p>
          </div>

          <div className="muted" style={{ fontSize: 13 }}>
            Tip: leave both shot type and zone on “All” first, then narrow down.
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
            Team
            <select
              value={team}
              onChange={(e) => {
                setTeam(e.target.value);
                setSelectedPreset("ALL");
              }}
              disabled={!meta?.teams?.length}
            >
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {normalizeLabel(t, meta?.teamNames)}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select
              value={opp}
              onChange={(e) => {
                setOpp(e.target.value);
                setSelectedPreset("ALL");
              }}
              disabled={!meta?.teams?.length}
            >
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {normalizeLabel(t, meta?.teamNames)}
                </option>
              ))}
            </select>
          </label>

          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="btn btn--secondary" type="button" onClick={swapTeams} disabled={!team || !opp}>
              Swap matchup ↔
            </button>
          </div>

          <label>
            Shot type (optional)
            <select
              value={shotType}
              onChange={(e) => {
                setShotType(e.target.value);
                setSelectedPreset("ALL");
              }}
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
            <select
              value={zone}
              onChange={(e) => {
                setZone(e.target.value);
                setSelectedPreset("ALL");
              }}
              disabled={!meta?.zones?.length}
            >
              <option value="">All</option>
              {(meta?.zones ?? []).map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </label>

          <label>
            Max shots (sampling cap)
            <input
              type="number"
              min={1000}
              max={100000}
              step={1000}
              value={maxShots}
              onChange={(e) => setMaxShots(clamp(Number(e.target.value || "0"), 1000, 100000))}
            />
            <div className="help">Large values keep more shots but may render slower.</div>
          </label>
        </div>

        {sameTeamSelected ? (
          <p className="error" style={{ marginTop: 12 }}>
            Team and opponent cannot be the same. Choose a real matchup.
          </p>
        ) : null}

        {error ? (
          <p className="error" style={{ marginTop: 12 }}>
            {error}
          </p>
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
            <h2 style={{ marginBottom: 6 }}>Heatmap Output</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              {subtitle}
            </p>
          </div>

          {heatmap?.image_base64 ? (
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
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
                  try {
                    await navigator.clipboard.writeText(subtitle);
                  } catch {
                    // ignore
                  }
                }}
              >
                Copy Summary
              </button>
            </div>
          ) : null}
        </div>

        {!heatmap?.image_base64 ? (
          <div
            style={{
              borderRadius: 18,
              padding: "28px 18px",
              marginTop: 12,
              background: "rgba(255,255,255,0.03)",
              border: "1px dashed rgba(255,255,255,0.14)",
              textAlign: "center",
              color: "rgba(255,255,255,0.7)",
            }}
          >
            {loading ? "Rendering heatmap…" : "No heatmap loaded yet. Click Render Heatmap to generate one."}
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
              <StatCard label="Rendered Matchup" value={`${team} vs ${opp}`} accent />
              <StatCard label="Rendered Type" value={shotType || "All"} />
              <StatCard label="Rendered Zone" value={zone || "All"} />
              <StatCard label="Endpoint Used" value={heatmap._endpoint_used || "Backend API"} />
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
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${heatmap.image_base64}`}
                alt="Shot heatmap"
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
                This heatmap should change when the matchup, season, shot type, or zone changes.
                If two renders look identical, verify the chosen filters actually produce distinct subsets
                in the underlying shot dataset.
              </div>
            </div>

            {heatmap.caption ? (
              <div className="muted" style={{ fontSize: 13 }}>
                {heatmap.caption}
              </div>
            ) : null}

            {typeof heatmap.n_shots === "number" ? (
              <div className="muted" style={{ fontSize: 13 }}>
                Shots used in render: {heatmap.n_shots.toLocaleString()}
              </div>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}