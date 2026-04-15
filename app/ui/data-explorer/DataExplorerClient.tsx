// app/ui/data-explorer/DataExplorerClient.tsx
//
// Data Explorer UI subsystem component
// Extracted from the page for better architecture organization

"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  fetchMetaOptions,
  fetchPipelineInfo,
  fetchTeamPlaytypesPreview,
  getTeamPlaytypesCsvUrl,
} from "../../services/dataExplorer";

type MetaOptions = {
  seasons: string[];
  teams: string[];
  teamNames?: Record<string, string>;
  playTypes?: string[];
  sides?: string[];
  hasMlPredictions?: boolean;
  _fallback?: boolean;
};

type PipelineInfo = {
  dataSource?: string;
  etl?: any;
  cleaning_and_aggregation?: string[];
  modeling?: string[];
};

type TeamPlayRow = Record<string, any>;

function fmtNum(v: any, digits = 3) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function fmtInt(v: any) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return String(Math.round(n));
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// localStorage can throw in some environments (private mode / blocked storage),
// so I wrap reads/writes to keep the page from crashing over a preference.
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
    // If storage is blocked, I just skip saving prefs. Not worth breaking the page.
  }
}

export default function DataExplorerClient() {
  const [meta, setMeta] = useState<MetaOptions>({
    seasons: [],
    teams: [],
    playTypes: [],
    sides: ["offense", "defense"],
    hasMlPredictions: false,
  });

  const [pipeline, setPipeline] = useState<PipelineInfo | null>(null);

  const [season, setSeason] = useState<string>("");
  const [team, setTeam] = useState<string>("");
  const [side, setSide] = useState<string>("offense");
  const [playType, setPlayType] = useState<string>(""); // blank means “all play types”
  const [minPoss, setMinPoss] = useState<number>(0);
  const [limit, setLimit] = useState<number>(200);
  const [rows, setRows] = useState<TeamPlayRow[]>([]);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [returnedRows, setReturnedRows] = useState<number>(0);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // UI-only presentation controls
  const [sortKey, setSortKey] = useState<string>("PPP");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [search, setSearch] = useState<string>("");
  const [showPipeline, setShowPipeline] = useState<boolean>(true);

  // Prevents stale responses when filters change rapidly.
  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const m = await fetchMetaOptions();
        if (cancelled) return;

        setMeta(m);

        // Sensible defaults so the page loads with data:
        // - most recent season
        // - TOR if available (nice for demos), otherwise first team
        const seasons = m.seasons ?? [];
        const teams = m.teams ?? [];

        const defaultSeason = seasons.length ? seasons[seasons.length - 1] : "";
        const defaultTeam = teams.includes("TOR") ? "TOR" : teams[0] ?? "";

        setSeason((prev) => prev || defaultSeason);
        setTeam((prev) => prev || defaultTeam);

        // Pipeline info is “nice to have” for reviewers.
        const p = await fetchPipelineInfo();
        if (!cancelled) setPipeline(p);
      } catch (e: any) {
        console.error(e);
        if (!cancelled) {
          setError(e?.message ?? "Failed to load metadata.");
        }
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);
  // Restore UI prefs (purely convenience)
  // This does NOT change the API itself — it only restores how I was viewing the table last time.
  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;

    const raw = safeLocalGet("nbaPlayRanker_dataExplorer_v1");
    if (!raw) return;

    try {
      const p = JSON.parse(raw);
      const sk = String(p.sortKey ?? "PPP");
      const sd = String(p.sortDir ?? "desc") as "asc" | "desc";
      const sp = String(p.search ?? "");
      const lim = Number(p.limit ?? limit);
      const mp = Number(p.minPoss ?? minPoss);
      const pl = String(p.playType ?? "");
      const si = String(p.side ?? side);
      const show = Boolean(p.showPipeline ?? true);

      if (sd === "asc" || sd === "desc") setSortDir(sd);
      setSortKey(sk);
      setSearch(sp);
      if (Number.isFinite(lim)) setLimit(lim);
      if (Number.isFinite(mp)) setMinPoss(mp);
      if (si) setSide(si);
      setPlayType(pl);
      setShowPipeline(show);
    } catch {
      // If prefs are corrupted, I ignore them and fall back to defaults.
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save UI prefs any time the user changes view settings (again: UI-only).
  useEffect(() => {
    safeLocalSet(
      "nbaPlayRanker_dataExplorer_v1",
      JSON.stringify({
        sortKey,
        sortDir,
        search,
        limit,
        minPoss,
        playType,
        side,
        showPipeline,
      })
    );
  }, [sortKey, sortDir, search, limit, minPoss, playType, side, showPipeline]);
  // Fetch when API-facing filters change; UI search/sort is client-side only.
  useEffect(() => {
    let cancelled = false;

    async function run() {

      if (!season) return;

      const myId = ++requestIdRef.current;

      try {
        setLoading(true);
        setError(null);

        const resp = await fetchTeamPlaytypesPreview({
          season,
          team: team || undefined, // if blank, I treat it as “all teams”
          side: side || undefined,
          playType: playType || undefined,
          minPoss,
          limit,
        });

        if (cancelled) return;
        if (requestIdRef.current !== myId) return;

        setTotalRows(Number(resp.total_rows ?? 0));
        setReturnedRows(Number(resp.returned_rows ?? 0));
        setRows(Array.isArray(resp.rows) ? resp.rows : []);
      } catch (e: any) {
        console.error(e);
        if (!cancelled) {
          setError(e?.message ?? "Unable to load data.");
          setRows([]);
          setTotalRows(0);
          setReturnedRows(0);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [season, team, side, playType, minPoss, limit]);
  const csvUrl = useMemo(() => {
    return getTeamPlaytypesCsvUrl({
      season: season || undefined,
      team: team || undefined,
      side: side || undefined,
      playType: playType || undefined,
      minPoss,
      limit: Math.max(limit, 1000),
    });
  }, [season, team, side, playType, minPoss, limit]);

  const columns = useMemo(() => {
    const base = [
      { key: "PLAY_TYPE", label: "Play Type" },
      { key: "SIDE", label: "Side" },
      { key: "POSS", label: "Possessions" },
      { key: "POSS_PCT", label: "Usage %" },
      { key: "PPP", label: "PPP" },
      { key: "EFG_PCT", label: "eFG%" },
      { key: "TOV_POSS_PCT", label: "TOV%" },
      { key: "FT_POSS_PCT", label: "FT%" },
    ];

    const keysInData = new Set(Object.keys(rows?.[0] ?? {}));
    return base.filter((c) => keysInData.has(c.key));
  }, [rows]);

  // Helper label so the header reads nicely (“All teams” vs “TOR (Raptors)”)
  const teamLabel = useMemo(() => {
    if (!team) return "All teams";
    const name = meta.teamNames?.[team];
    return name ? `${team} (${name})` : team;
  }, [meta.teamNames, team]);
  // Client-side search + sort on already-fetched rows.
  const viewRows = useMemo(() => {
    const list = Array.isArray(rows) ? rows : [];
    const q = search.trim().toLowerCase();

    const filtered = q
      ? list.filter((r) => {
          const pt = String(r.PLAY_TYPE ?? "").toLowerCase();
          const sd = String(r.SIDE ?? "").toLowerCase();
          const tm = String(r.TEAM_ABBREVIATION ?? "").toLowerCase();
          return pt.includes(q) || sd.includes(q) || tm.includes(q);
        })
      : list;

    const key = sortKey;
    const dir = sortDir;

    const sorted = [...filtered].sort((a, b) => {
      const av = a?.[key];
      const bv = b?.[key];

      const an = Number(av);
      const bn = Number(bv);

      const bothNum = Number.isFinite(an) && Number.isFinite(bn);
      if (bothNum) return dir === "asc" ? an - bn : bn - an;

      const as = String(av ?? "");
      const bs = String(bv ?? "");
      return dir === "asc" ? as.localeCompare(bs) : bs.localeCompare(as);
    });

    return sorted;
  }, [rows, search, sortKey, sortDir]);
  // KPI tiles (based on what’s currently visible after UI filtering)
  // These are just quick sanity-check stats for the preview.
  const kpis = useMemo(() => {
    const list = viewRows;
    const poss = list.map((r) => Number(r.POSS)).filter((x) => Number.isFinite(x));
    const ppp = list.map((r) => Number(r.PPP)).filter((x) => Number.isFinite(x));

    const sumPoss = poss.reduce((a, b) => a + b, 0);
    const avgPPP = ppp.length ? ppp.reduce((a, b) => a + b, 0) / ppp.length : NaN;
    const maxPPP = ppp.length ? Math.max(...ppp) : NaN;
    const minPPP = ppp.length ? Math.min(...ppp) : NaN;

    return {
      sumPoss,
      avgPPP,
      maxPPP,
      minPPP,
    };
  }, [viewRows]);
  // PPP quality bar (visual only)
  const pppBar = useMemo(() => {
    const v = Number(kpis.avgPPP);
    if (!Number.isFinite(v)) return { pct: 0, label: "\u2014" };
    // Typical PPP range ~[0.7 .. 1.3]
    const pct = clamp01((v - 0.7) / 0.6) * 100;
    return { pct, label: fmtNum(v, 3) };
  }, [kpis.avgPPP]);

  const heroStyle: React.CSSProperties = {
    borderRadius: 18,
    padding: "18px 18px 14px",
    background:
      "linear-gradient(135deg, rgba(56,189,248,0.16), rgba(99,102,241,0.14), rgba(34,197,94,0.10))",
    border: "1px solid rgba(255,255,255,0.10)",
  };

  // Reproducible API URL for the current filters.
  const apiExample = useMemo(() => {
    const parts: string[] = [];
    parts.push(`${API_BASE}/data/team-playtypes?season=${season}`);
    if (team) parts.push(`team=${team}`);
    if (side) parts.push(`side=${side}`);
    if (playType) parts.push(`play_type=${encodeURIComponent(playType)}`);
    if (minPoss) parts.push(`min_poss=${minPoss}`);
    if (limit) parts.push(`limit=${limit}`);
    const [first, ...rest] = parts;
    return rest.length ? `${first}&${rest.join("&")}` : first;
  }, [season, team, side, playType, minPoss, limit]);

  return (
    <main className="page" style={{ paddingBottom: 56 }}>
      {/* HERO */}
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
              Data Explorer
            </h1>
            <p className="muted" style={{ marginTop: 6, fontSize: 14, marginBottom: 0 }}>
              Preview the <strong>cleaned, team-level dataset</strong> used by the recommenders. This page contains{" "}
              <strong>no predictions</strong>—only filterable data you can export for analysis.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <a className="btn btn--secondary" href={csvUrl} target="_blank" rel="noopener noreferrer">
              Export CSV
            </a>
            <Link className="btn" href="/matchup">
              Next: Baseline Matchup
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
          <p className="muted" style={{ fontSize: 12, margin: 0 }}>
            Season <strong>{season || "—"}</strong> • <strong>{teamLabel}</strong> • Side{" "}
            <strong>{side}</strong> • Showing <strong>{returnedRows}</strong> / <strong>{totalRows}</strong>{" "}
            rows
          </p>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button
              className="btn btn--secondary"
              type="button"
              onClick={() => {
                // Reset is intentionally UI-only — I’m just putting the controls back to a clean demo state.
                setSearch("");
                setSortKey("PPP");
                setSortDir("desc");
                setMinPoss(0);
                setPlayType("");
                setTeam(meta.teams?.includes("TOR") ? "TOR" : meta.teams?.[0] ?? "");
                setSide("offense");
                setLimit(200);
              }}
              title="Resets UI filters (does not change backend behavior)"
              disabled={loading}
            >
              Reset UI
            </button>
          </div>
        </div>
      </header>

      {/* FILTERS */}
      <section className="card" style={{ marginTop: 14 }}>
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
            <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>Filters</h2>
            <p className="muted" style={{ fontSize: 13, margin: 0 }}>
              Filter the dataset preview. Export uses the same filters (with a higher row cap).
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Search (UI)
              <input
                className="input"
                style={{ width: 260 }}
                placeholder="Play type / side / team…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>

            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Sort
              <select
                className="input"
                style={{ width: 150 }}
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value)}
                disabled={!columns.length}
              >
                {columns.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              className="btn btn--secondary"
              type="button"
              onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
              disabled={!columns.length}
              title="Toggle sort direction (UI only)"
            >
              {sortDir === "asc" ? "Asc ↑" : "Desc ↓"}
            </button>
          </div>
        </div>

        <form className="form-grid" onSubmit={(e) => e.preventDefault()} style={{ marginTop: 10 }}>
          <label>
            Season
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              {(meta.seasons?.length ? meta.seasons : []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Team (optional)
            <select className="input" value={team} onChange={(e) => setTeam(e.target.value)}>
              <option value="">All teams</option>
              {(meta.teams?.length ? meta.teams : []).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label>
            Side
            <select className="input" value={side} onChange={(e) => setSide(e.target.value)}>
              {(meta.sides?.length ? meta.sides : ["offense", "defense"]).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Play Type (optional)
            <select className="input" value={playType} onChange={(e) => setPlayType(e.target.value)}>
              <option value="">All play types</option>
              {(meta.playTypes?.length ? meta.playTypes : []).map((pt) => (
                <option key={pt} value={pt}>
                  {pt}
                </option>
              ))}
            </select>
          </label>

          <label>
            Min possessions
            <input
              className="input"
              type="number"
              min={0}
              step={1}
              value={minPoss}
              onChange={(e) => setMinPoss(Number(e.target.value))}
            />
          </label>

          <label>
            Rows to show
            <input
              className="input"
              type="number"
              min={1}
              max={1000}
              step={1}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>
        </form>

        {/* KPI strip (visual helpers only) */}
        <div className="grid" style={{ marginTop: 14 }}>
          <div className="kpi">
            <div className="label">Visible rows</div>
            <div className="value">{fmtInt(viewRows.length)}</div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              After UI search + sort.
            </div>
          </div>

          <div className="kpi">
            <div className="label">Total possessions</div>
            <div className="value">{fmtInt(kpis.sumPoss)}</div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Sum over visible rows.
            </div>
          </div>

          <div className="kpi">
            <div className="label">Avg PPP</div>
            <div className="value">{Number.isFinite(Number(kpis.avgPPP)) ? fmtNum(kpis.avgPPP, 3) : "—"}</div>
            <div
              style={{
                marginTop: 10,
                height: 10,
                borderRadius: 999,
                background: "rgba(15,23,42,0.08)",
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.10)",
              }}
              title="Visual indicator only (UI)"
            >
              <div
                style={{
                  width: `${pppBar.pct}%`,
                  height: "100%",
                  background:
                    "linear-gradient(90deg, rgba(59,130,246,0.60), rgba(34,197,94,0.35))",
                }}
              />
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Typical range ~0.7–1.3 PPP.
            </div>
          </div>

          <div className="kpi">
            <div className="label">PPP range</div>
            <div className="value">
              {Number.isFinite(Number(kpis.minPPP)) ? fmtNum(kpis.minPPP, 3) : "—"} –{" "}
              {Number.isFinite(Number(kpis.maxPPP)) ? fmtNum(kpis.maxPPP, 3) : "—"}
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Min/Max within visible rows.
            </div>
          </div>
        </div>

        {/* Loading / Error states */}
        {loading && (
          <p className="muted" style={{ marginTop: 14 }}>
            Loading dataset preview…
          </p>
        )}

        {error && !loading && (
          <p className="muted" style={{ marginTop: 14 }}>
            {error}
          </p>
        )}

        {!loading && !error && rows.length === 0 && (
          <p className="muted" style={{ marginTop: 14 }}>
            No rows match these filters.
          </p>
        )}
      </section>

      {/* TABLE */}
      <section className="card">
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
            <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>Preview table</h2>
            <p className="muted" style={{ fontSize: 13, margin: 0 }}>
              Committee-friendly subset of columns. (UI search/sort does not change the exported CSV.)
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <span className="badge">
              Export rows cap: {Math.max(limit, 1000)}
            </span>
            <a className="btn" href={csvUrl} target="_blank" rel="noopener noreferrer">
              Export CSV
            </a>
          </div>
        </div>

        {!loading && !error && viewRows.length > 0 ? (
          <div style={{ marginTop: 10, overflowX: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  {columns.map((c) => (
                    <th
                      key={c.key}
                      style={{ cursor: "pointer" }}
                      title="Click to sort (UI)"
                      onClick={() => {
                        // Sorting here is only on the already-fetched rows — it doesn’t change what the API returns.
                        if (sortKey === c.key) {
                          setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                        } else {
                          setSortKey(c.key);
                          setSortDir("desc");
                        }
                      }}
                    >
                      {c.label}{" "}
                      {sortKey === c.key ? (
                        <span className="muted" style={{ fontSize: 11 }}>
                          {sortDir === "asc" ? "↑" : "↓"}
                        </span>
                      ) : null}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {viewRows.map((r, idx) => (
                  <tr key={`${r.SEASON}-${r.TEAM_ABBREVIATION}-${r.SIDE}-${r.PLAY_TYPE}-${idx}`}>
                    <td>{idx + 1}</td>

                    {columns.map((c) => {
                      const v = r[c.key];

                      // Fields that should read as integers in the table.
                      if (c.key === "POSS" || c.key === "GP") {
                        return <td key={c.key}>{Number.isFinite(Number(v)) ? fmtInt(v) : "—"}</td>;
                      }

                      // Percent-ish columns: I render as % for readability, but keep raw in the title tooltip.
                      if (c.key.endsWith("_PCT") || c.key === "POSS_PCT") {
                        const n = Number(v);
                        return (
                          <td key={c.key} title={Number.isFinite(n) ? String(n) : ""}>
                            {Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—"}
                          </td>
                        );
                      }

                      // Numeric fallback formatting.
                      if (typeof v === "number" || !isNaN(Number(v))) {
                        return <td key={c.key}>{fmtNum(v, 3)}</td>;
                      }

                      return <td key={c.key}>{String(v ?? "—")}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      {/* PIPELINE / DEFENSE PANEL */}
      <section className="card">
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
            <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>
              Pipeline summary (what this data represents)
            </h2>
            <p className="muted" style={{ fontSize: 13, margin: 0 }}>
              Evidence for the defense: cleaning + aggregation pipeline (player rows → team rows) with export parity.
            </p>
          </div>

          <button
            className="btn btn--secondary"
            type="button"
            onClick={() => setShowPipeline((v) => !v)}
          >
            {showPipeline ? "Hide details" : "Show details"}
          </button>
        </div>

        {showPipeline ? (
          <>
            <div style={{ marginTop: 10, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <span className="badge blue">No predictions on this page</span>
              <span className="badge">Filter + export only</span>
              <span className="badge">Team-level aggregation</span>
            </div>

            <p className="muted" style={{ fontSize: 13, marginTop: 10 }}>
              {pipeline?.dataSource ? pipeline.dataSource : "Data source: (not loaded)"}{" "}
              <span className="badge">Explainability</span>
            </p>

            <ul className="muted" style={{ fontSize: 13, paddingLeft: 18, marginTop: 8 }}>
              {(pipeline?.cleaning_and_aggregation ?? [
                "Aggregate player rows into team-level play-type rows (possession-weighted).",
                "Recompute team-level usage % (POSS_PCT) to avoid double-counting.",
                "Add reliability weights to reduce small-sample noise (shrinkage).",
              ]).map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>

            <p className="muted" style={{ marginTop: 10, fontSize: 11 }}>
              API call:{" "}
              <code>{apiExample}</code>
            </p>
          </>
        ) : (
          <p className="muted" style={{ fontSize: 13, marginTop: 10 }}>
            Hidden. Toggle “Show details” for defense notes + API trace.
          </p>
        )}
      </section>
    </main>
  );
}
