"use client";
// Shot explorer page UI.

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  fetchMetaOptions,
  fetchPbpMetaOptions,
  fetchPbpShotsPreview,
  getPbpShotsCsvUrl,
} from "../../../services/shotAnalysis";

import type { PbpShotsPreviewResponse } from "../../../services/shotAnalysis";

type Meta = {
  seasons: string[];
  teams: string[];
  shotTypes: string[];
  zones: string[];
  teamNames?: Record<string, string>;
};

function pickDefaultSeason(seasons: string[]) {
  // Default to the most recent season available (or 2021-22 if meta is empty).
  if (!seasons?.length) return "2021-22";
  return seasons[seasons.length - 1];
}

function pickDefaultTeam(teams: string[], prefer: string) {
  // Prefer a familiar team (TOR) when available so the first load feels “not blank.”
  if (!teams?.length) return prefer;
  return teams.includes(prefer) ? prefer : teams[0];
}

function chip(text: string) {
  // Small “filter summary” badges so you can screenshot the exact query context for defense/docs.
  return (
    <span
      className="badge"
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.10)",
      }}
    >
      {text}
    </span>
  );
}

export default function ShotExplorerPage() {
  const [meta, setMeta] = useState<Meta | null>(null);

  // Filter inputs (these map directly to the /pbp/shots/preview query params).
  const [season, setSeason] = useState<string>("");
  const [team, setTeam] = useState<string>("TOR");
  const [opp, setOpp] = useState<string>("");
  const [shotType, setShotType] = useState<string>("");
  const [zone, setZone] = useState<string>("");
  const [limit, setLimit] = useState<number>(50);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [data, setData] = useState<PbpShotsPreviewResponse | null>(null);

  // Load meta options once:
  // - Dataset2 provides the “real” dropdown values (seasons/teams/shotTypes/zones).
  // - Dataset1 provides teamNames (nice labels like "TOR (Raptors)").
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [mMain, mPbp] = await Promise.all([
          fetchMetaOptions(),
          fetchPbpMetaOptions(),
        ]);

        const merged: Meta = {
          seasons: mPbp.seasons,
          teams: mPbp.teams,
          shotTypes: mPbp.shotTypes ?? [],
          zones: mPbp.zones ?? [],
          teamNames: mMain.teamNames ?? {},
        };

        if (cancelled) return;
        setMeta(merged);

        // Auto-fill sensible defaults so the user can “Run” immediately without touching everything.
        setSeason((prev) => prev || pickDefaultSeason(merged.seasons));
        setTeam((prev) => prev || pickDefaultTeam(merged.teams, "TOR"));
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? "Failed to load meta options.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const teamLabel = useMemo(() => {
    // Friendly label for headers + chips (abbr + long name when available).
    const name = meta?.teamNames?.[team];
    return name ? `${team} (${name})` : team;
  }, [meta?.teamNames, team]);

  const oppLabel = useMemo(() => {
    // Opponent label is only relevant when opp filter is set.
    if (!opp) return "";
    const name = meta?.teamNames?.[opp];
    return name ? `${opp} (${name})` : opp;
  }, [meta?.teamNames, opp]);

  const csvUrl = useMemo(() => {
    // CSV export is for “heavy lifting”, so we export up to 5,000 rows.
    // Preview limit stays smaller to keep UI fast.
    if (!season || !team) return "";
    return getPbpShotsCsvUrl({
      season,
      team,
      opp: opp || undefined,
      shotType: shotType || undefined,
      zone: zone || undefined,
      limit: 5000,
    });
  }, [season, team, opp, shotType, zone]);

  async function run() {
    // Minimum required filters so the backend isn’t asked for “everything.”
    if (!season || !team) {
      setError("Please select at least season and team.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setData(null);

      // This hits /pbp/shots/preview and returns columns + a small row sample.
      const res = await fetchPbpShotsPreview({
        season,
        team,
        opp: opp || undefined,
        shotType: shotType || undefined,
        zone: zone || undefined,
        limit,
      });

      setData(res);
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch shots.");
    } finally {
      setLoading(false);
    }
  }

  // Table-building helpers (backend returns canonical column list + row dicts).
  const columns = data?.columns ?? [];
  const rows = data?.rows ?? [];

  const summaryChips = useMemo(() => {
    // Visual “receipt” of the query settings (useful for screenshots + clarity).
    const chips: React.ReactNode[] = [];
    if (season) chips.push(chip(`Season: ${season}`));
    if (team) chips.push(chip(`Team: ${team}`));
    if (opp) chips.push(chip(`Opp: ${opp}`));
    if (shotType) chips.push(chip(`ShotType: ${shotType}`));
    if (zone) chips.push(chip(`Zone: ${zone}`));
    chips.push(chip(`Preview: ${limit} rows`));
    return chips;
  }, [season, team, opp, shotType, zone, limit]);

  return (
    <main className="page">
      {/* Hero header: sets context + points users to the recommendation page */}
      <header
        className="card"
        style={{
          padding: 20,
          borderRadius: 16,
          background:
            "linear-gradient(135deg, rgba(34,197,94,0.15), rgba(59,130,246,0.12) 40%, rgba(15,22,44,0.95))",
          border: "1px solid rgba(255,255,255,0.10)",
        }}
      >
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <h1 style={{ margin: 0 }}>Shots Explorer</h1>
          <span className="badge blue">Dataset2 • Raw PBP</span>
          <span className="badge">One row per shot</span>
        </div>

        <p className="muted" style={{ marginTop: 8 }}>
          Browse raw shots from the play-by-play dataset. Filter, preview quickly, then export CSV
          for deeper analysis in Excel / R / Python.
        </p>

        <p className="muted" style={{ margin: 0 }}>
          Want recommendations? See <Link href="/shot-plan">Shot Plan</Link>.
        </p>
      </header>

      {/* Filters: all user inputs + actions live here */}
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>Filters</h2>

          {/* Actions: Run = preview query, CSV = export query */}
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <button className="btn primary" onClick={run} disabled={loading}>
              {loading ? "Loading…" : "Run"}
            </button>

            {csvUrl ? (
              <a
                className="btn"
                href={csvUrl}
                target="_blank"
                rel="noreferrer"
                style={{
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.12)",
                }}
              >
                Export CSV (up to 5,000 rows)
              </a>
            ) : null}
          </div>
        </div>

        {/* Primary filters: season + team + optional opponent */}
        <div className="grid" style={{ marginTop: 12, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          <div>
            <div className="label">Season</div>
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              {(meta?.seasons ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="label">Team</div>
            <select className="input" value={team} onChange={(e) => setTeam(e.target.value)}>
              {(meta?.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {meta?.teamNames?.[t] ? `${t} (${meta.teamNames[t]})` : t}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="label">Opponent (optional)</div>
            <select className="input" value={opp} onChange={(e) => setOpp(e.target.value)}>
              <option value="">All</option>
              {(meta?.teams ?? [])
                .filter((t) => t !== team)
                .map((t) => (
                  <option key={t} value={t}>
                    {meta?.teamNames?.[t] ? `${t} (${meta.teamNames[t]})` : t}
                  </option>
                ))}
            </select>
          </div>
        </div>

        {/* Secondary filters: optional shotType + zone + preview limit */}
        <div className="grid" style={{ marginTop: 12, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
          <div>
            <div className="label">Shot type (optional)</div>
            <select className="input" value={shotType} onChange={(e) => setShotType(e.target.value)}>
              <option value="">All</option>
              {(meta?.shotTypes ?? []).map((st) => (
                <option key={st} value={st}>
                  {st}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="label">Zone (optional)</div>
            <select className="input" value={zone} onChange={(e) => setZone(e.target.value)}>
              <option value="">All</option>
              {(meta?.zones ?? []).map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="label">Preview limit</div>
            <input
              className="input"
              type="number"
              min={10}
              max={500}
              step={10}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value || "0"))}
            />
            <div className="muted" style={{ marginTop: 6 }}>
              Preview only. Use CSV export for more rows.
            </div>
          </div>
        </div>

        {/* Chips act like a “receipt” of the current filter state */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
          {summaryChips}
        </div>

        {/* Error panel keeps failures obvious without breaking the page */}
        {error ? (
          <div className="card" style={{ marginTop: 12, borderColor: "rgba(239,68,68,0.35)" }}>
            <div className="muted" style={{ color: "rgba(239,68,68,0.95)" }}>
              {error}
            </div>
          </div>
        ) : null}
      </section>

      {/* Preview results: intentionally small to keep performance stable */}
      <section className="card">
        <h2>Preview</h2>

        {/* Header chips show the final interpreted query (nice for screenshots) */}
        <p className="muted" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {chip(teamLabel)}
          {season ? chip(season) : null}
          {opp ? chip(`vs ${oppLabel || opp}`) : chip("All opponents")}
          {shotType ? chip(shotType) : chip("All shot types")}
          {zone ? chip(zone) : chip("All zones")}
        </p>

        {/* Empty state: tells you what to do next */}
        {!data && !loading ? (
          <div className="card" style={{ background: "rgba(255,255,255,0.03)" }}>
            <div className="muted">
              Run a query to see rows. This preview is intentionally small for speed.
            </div>
          </div>
        ) : null}

        {/* Loading state: keeps page from looking “frozen” */}
        {loading ? (
          <div className="card" style={{ background: "rgba(255,255,255,0.03)" }}>
            <div className="muted">Loading preview…</div>
          </div>
        ) : null}

        {/* Results state: show row counts + table */}
        {data ? (
          <>
            {/* Backend returns both returned_rows (preview) and total_rows (matched) */}
            <div className="card" style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
              {chip(`Returned: ${data.returned_rows}`)}
              {chip(`Total matched: ${data.total_rows}`)}
              {data.shot_type ? chip(`shot_type=${data.shot_type}`) : null}
              {data.zone ? chip(`zone=${data.zone}`) : null}
            </div>

            {/* Table: uses server-sent column list so UI doesn’t break if schema changes */}
            <div className="card" style={{ padding: 0, overflowX: "auto", marginTop: 12 }}>
              <table className="table">
                <thead>
                  <tr>
                    {columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, idx) => (
                    <tr key={idx}>
                      {columns.map((c) => (
                        <td key={c}>{String((r as any)[c] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="muted" style={{ marginTop: 10 }}>
              Showing {rows.length} row(s). Columns follow the canonical Dataset2 shot schema.
            </p>
          </>
        ) : null}
      </section>
    </main>
  );
}
