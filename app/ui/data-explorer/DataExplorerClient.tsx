// app/ui/data-explorer/DataExplorerClient.tsx
//
// Data Explorer UI subsystem component
// Extracted from the page for better architecture organization

"use client";

import Link from "next/link";
import styles from "./AnalystWorkspace.module.css";
import { useEffect, useMemo, useRef, useState } from "react";
import DataQualityPanel from "./DataQualityPanel";
import DatasetReloadPanel from "./DatasetReloadPanel";
import type { ReloadDataset } from "../../domain/dataset-reload";
import DatasetMetadataPanel from "./DatasetMetadataPanel";
import ShotDataExplorer from "./ShotDataExplorer";
import AnalystSeasonSummary from "./AnalystSeasonSummary";
import SavedQueriesPanel from "./SavedQueriesPanel";
import type { FilterOptions, QueryDefinition, SavedQuery } from "../../domain/saved-query";
import {
  API_BASE,
  authenticatedDownload,
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

type ExplorerDataset = "synergy" | "shots";
const DATASET_KEY = "nbaPlayRanker_dataExplorer_dataset";

// SCRUM-627: switch between the Synergy play-type data and the play-by-play shots.
function DatasetSwitch({ value, onChange }: { value: ExplorerDataset; onChange: (d: ExplorerDataset) => void }) {
  const option = (id: ExplorerDataset, label: string) => (
    <button
      type="button"
      role="radio"
      aria-checked={value === id}
      className={value === id ? "btn" : "btn btn--secondary"}
      style={value === id ? { background: "#0f172a", color: "#fff", borderColor: "#0f172a" } : undefined}
      onClick={() => onChange(id)}
    >
      {label}
    </button>
  );
  return (
    <div role="radiogroup" aria-label="Dataset" style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
      {option("synergy", "Synergy play types")}
      {option("shots", "Play-by-play shots")}
    </div>
  );
}

export default function DataExplorerClient() {
  const [dataset, setDataset] = useState<ExplorerDataset>("synergy");
  const [pendingReload, setPendingReload] = useState<ReloadDataset | null>(null);
  const [synergyRevision, setSynergyRevision] = useState(0);
  const [shotsRevision, setShotsRevision] = useState(0);
  const [metadataRevision, setMetadataRevision] = useState(-1);
  const [metadataError, setMetadataError] = useState("");
  const loadedMetadata = useRef(false);
  const synergyReloading = pendingReload === "synergy";
  const synergyBlocked = synergyReloading || metadataRevision !== synergyRevision;
  function afterReload(target: ReloadDataset) {
    if (target === "synergy") setSynergyRevision(value => value + 1);
    else setShotsRevision(value => value + 1);
  }
  function updatePending(target: ReloadDataset | null) {
    setPendingReload(target);
    if (target === "synergy") {
      requestIdRef.current++;
      setMetadataRevision(-1);
      setRows([]);
      setTotalRows(0);
      setReturnedRows(0);
    }
  }
  useEffect(() => {
    if (safeLocalGet(DATASET_KEY) === "shots") setDataset("shots");
  }, []);
  function switchDataset(next: ExplorerDataset) {
    setDataset(next);
    safeLocalSet(DATASET_KEY, next);
  }
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
  const [side, setSide] = useState<string>("");
  const [playType, setPlayType] = useState<string>(""); // blank means “all play types”
  const [minPoss, setMinPoss] = useState<number>(0);
  const [limit, setLimit] = useState<number>(200);
  const [rows, setRows] = useState<TeamPlayRow[]>([]);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [returnedRows, setReturnedRows] = useState<number>(0);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  // UI-only presentation controls
  const [sortKey, setSortKey] = useState<string>("PPP");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [search, setSearch] = useState<string>("");

  // Prevents stale responses when filters change rapidly.
  const requestIdRef = useRef(0);
  const didInitRef = useRef(false);
  useEffect(() => {
    let cancelled = false;
    setMetadataRevision(-1);
    setMetadataError("");
    setPipeline(null);
    if (synergyReloading) return;
    // Pipeline notes are optional and must not hold up the refreshed table.
    void fetchPipelineInfo().then(value => { if (!cancelled) setPipeline(value); }).catch(() => {});
    async function init() {
      try {
        const m = await fetchMetaOptions();
        if (cancelled) return;
        if (m._fallback) throw new Error("Play-type options could not be refreshed.");
        // SCRUM-841: keep the analyst's own choices if they still exist; never pre-select one.
        const seasons = m.seasons ?? [], teams = m.teams ?? [];
        const sides = m.sides?.length ? m.sides : ["offense", "defense"];
        setMeta(m);
        setSeason(previous => seasons.includes(previous) ? previous : "");
        setTeam(previous => teams.includes(previous) ? previous : "");
        setSide(previous => sides.includes(previous) ? previous : "");
        setPlayType(previous => m.playTypes?.includes(previous) ? previous : "");
        loadedMetadata.current = true;
        setMetadataRevision(synergyRevision);
      } catch {
        if (!cancelled) setMetadataError("Could not refresh the play-type view. Retry the view below; another dataset reload is not needed.");
      }
    }
    void init();
    return () => { cancelled = true; };
  }, [synergyRevision, synergyReloading]);
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

      if (sd === "asc" || sd === "desc") setSortDir(sd);
      setSortKey(sk);
      setSearch(sp);
      if (Number.isFinite(lim)) setLimit(lim);
      if (Number.isFinite(mp)) setMinPoss(mp);
      if (si) setSide(si);
      setPlayType(pl);
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
      })
    );
  }, [sortKey, sortDir, search, limit, minPoss, playType, side]);
  // Fetch when API-facing filters change; UI search/sort is client-side only.
  useEffect(() => {
    let cancelled = false;

    async function run() {

      setRows([]);
      setTotalRows(0);
      setReturnedRows(0);
      if (!season || synergyBlocked) { setLoading(false); return; }

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
        if (!cancelled && requestIdRef.current === myId) {
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
  }, [season, team, side, playType, minPoss, limit, synergyRevision, synergyBlocked]);
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

  async function exportCsv() {
    if (!season || exporting || synergyBlocked || loading) return;
    setExporting(true);
    setExportError(null);
    try {
      await authenticatedDownload(
        csvUrl,
        `team_playtypes_${season}_${team || "all"}_${side || "all"}.csv`
      );
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Unable to export CSV. Please try again.");
    } finally {
      setExporting(false);
    }
  }

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
    if (synergyBlocked) return [];
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
  }, [rows, search, sortKey, sortDir, synergyBlocked]);
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

  // SCRUM-771: the current Synergy view as a saved query, and opening one back.
  const synergyQuery = useMemo<QueryDefinition>(
    () => ({
      dataset: "synergy",
      filters: { season, team, side, playType, minPoss, search },
      sortKey,
      sortDir,
      rowLimit: limit,
    }),
    [season, team, side, playType, minPoss, search, sortKey, sortDir, limit]
  );
  const synergyOptions = useMemo<FilterOptions>(
    () => ({ season: meta.seasons, team: meta.teams, playType: meta.playTypes }),
    [meta.seasons, meta.teams, meta.playTypes]
  );
  const [pendingShots, setPendingShots] = useState<SavedQuery | null>(null);
  function openSynergyQuery(query: SavedQuery) {
    if (query.dataset === "shots") {
      setPendingShots(query);
      switchDataset("shots");
      return;
    }
    if (dataset !== "synergy") switchDataset("synergy");
    const f = query.filters;
    setSeason(String(f.season ?? ""));
    setTeam(String(f.team ?? ""));
    setSide(String(f.side ?? ""));
    setPlayType(String(f.playType ?? ""));
    setMinPoss(Number(f.minPoss ?? 0) || 0);
    setSearch(String(f.search ?? ""));
    setSortKey(query.sortKey ?? "PPP");
    setSortDir(query.sortDir ?? "desc");
    setLimit(query.rowLimit);
  }

  if (dataset === "shots") {
    return (
      <div className={`page ${styles.workspace}`} style={{ paddingBottom: 56 }}>
        <header className={styles.header}>
          <div className={styles.eyebrow}>Analyst workspace</div>
          <h1 className="h1" style={{ margin: 0 }}>Data Explorer</h1>
          <p className="muted" style={{ marginTop: 6, fontSize: 14, marginBottom: 0 }}>
            Browse and export recorded shots, including regular-season and playoff games.
          </p>
          <DatasetSwitch value={dataset} onChange={switchDataset} />
        </header>
        <ShotDataExplorer initialQuery={pendingShots} onOpenOther={openSynergyQuery}
          refreshVersion={shotsRevision} reloading={pendingReload === "pbp"} />
        <details className={`card ${styles.tools}`}>
          <summary>Dataset tools</summary>
          <DatasetReloadPanel key="dataset-reload" onReloaded={afterReload} onPendingChange={updatePending} />
        </details>
      </div>
    );
  }

  return (
    <div className={`page ${styles.workspace}`} style={{ paddingBottom: 56 }}>
      {/* HERO */}
      <header className={styles.header}>
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
            <div className={styles.eyebrow}>Analyst workspace</div>
            <h1 className="h1" style={{ margin: 0 }}>
              Data Explorer
            </h1>
            <p className="muted" style={{ marginTop: 6, fontSize: 14, marginBottom: 0 }}>
              Browse recorded team play-type stats and export the selected data.
            </p>
            <DatasetSwitch value={dataset} onChange={switchDataset} />
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn btn--secondary" type="button" onClick={exportCsv} disabled={!season || loading || exporting || synergyBlocked} aria-busy={exporting}>
              {exporting ? "Exporting play-type rows…" : "Export play-type rows (CSV)"}
            </button>
            <Link className="btn" href="/model-metrics">
              Model Metrics
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
            Regular season <strong>{season || "—"}</strong> • <strong>{teamLabel}</strong> • Side{" "}
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
                setTeam("");
                setSide("");
                setLimit(200);
              }}
              title="Reset table and filter choices"
              disabled={loading}
            >
              Reset filters
            </button>
          </div>
        </div>
      </header>

      {exportError && <p role="alert" className="card">{exportError}</p>}
      {metadataError ? <section className="card" role="alert"><p>{metadataError}</p>
        <button className="btn btn--secondary" type="button" onClick={() => setSynergyRevision(value => value + 1)}>Retry view</button></section>
        : synergyBlocked ? <p className="card" role="status">{synergyReloading ? "Reloading play-type data…" : "Refreshing play-type options…"}</p> : null}
      {!synergyBlocked && <DataQualityPanel key={`quality-${synergyRevision}`} />}

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
              Choose a season, team and play type.
            </p>
          </div>

          <div className={styles.toolbar}>
            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Search this table
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
              title="Change table sort direction"
            >
              {sortDir === "asc" ? "Asc ↑" : "Desc ↓"}
            </button>
          </div>
        </div>

        <form className="form-grid" onSubmit={(e) => e.preventDefault()} style={{ marginTop: 10 }}>
          <label>
            Season
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              <option value="" disabled>Choose a season</option>
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
              <option value="">Both sides</option>
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
              Rows in this table.
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
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Average across visible play-type rows.
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
              CSV uses the selected filters, up to 1,000 rows. Table search and sorting do not change the download.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <span className="badge">
              CSV row limit: {Math.max(limit, 1000)}
            </span>
            <button className="btn" type="button" onClick={exportCsv} disabled={!season || loading || exporting || synergyBlocked} aria-busy={exporting}>
              {exporting ? "Exporting play-type rows…" : "Export play-type rows (CSV)"}
            </button>
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
                      title="Sort this table"
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

      <SavedQueriesPanel definition={synergyQuery} options={synergyOptions} onOpen={openSynergyQuery} />

      <AnalystSeasonSummary season={season} team={team} side={side} teams={meta.teams}
        teamNames={meta.teamNames} refreshVersion={synergyRevision} refreshing={synergyBlocked} />

      <details className={`card ${styles.tools}`}>
        <summary>Dataset tools and source</summary>
        <DatasetReloadPanel key="dataset-reload" onReloaded={afterReload} onPendingChange={updatePending} />
        {!synergyBlocked && <DatasetMetadataPanel key={`metadata-${synergyRevision}-${shotsRevision}`} />}
      </details>

      <details className="card">
        <summary>How this data is prepared</summary>
        <p className="muted">Player records are combined into team totals. The preview table shows recorded statistics, not model predictions.</p>
        <p className="muted">Source: {pipeline?.dataSource || "Unavailable"}</p>
        <ul className="muted">
          {(pipeline?.cleaning_and_aggregation ?? [
            "Combine player records into team play-type totals, weighted by possessions.",
            "Recalculate each play type's share of team possessions.",
            "Reduce the influence of small samples when preparing recommendations.",
          ]).map((step, index) => <li key={index}>{step}</li>)}
        </ul>
        <details><summary>API request</summary><code style={{ overflowWrap: "anywhere" }}>{apiExample}</code></details>
      </details>
    </div>
  );
}
