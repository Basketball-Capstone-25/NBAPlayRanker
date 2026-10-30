"use client";

// UI Layer — Data Explorer play-by-play shots (SCRUM-627).
// Analysts filter the shot dataset, page through 50 shots at a time and export the filtered view.

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  authenticatedDownload,
  canSearchShots,
  fetchPbpMetaOptions,
  fetchShotPage,
  SHOT_PAGE_SIZE,
  ShotAccessError,
  shotExportFilename,
  shotExportUrl,
  type PbpMetaOptions,
  type ShotFilters,
  type ShotPage,
  type ShotResult,
} from "../../services/shot-explorer";
import styles from "./ShotDataExplorer.module.css";

type Status = "idle" | "loading" | "ready" | "error";

const EMPTY: ShotFilters = { season: "", team: "", opp: "", shotType: "", zone: "", result: "" };

export default function ShotDataExplorer() {
  const [options, setOptions] = useState<PbpMetaOptions | null>(null);
  const [draft, setDraft] = useState<ShotFilters>(EMPTY);
  const [applied, setApplied] = useState<ShotFilters | null>(null);
  const [page, setPage] = useState(1);
  const [result, setResult] = useState<ShotPage | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [exporting, setExporting] = useState<"" | "csv" | "json">("");
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    let alive = true;
    fetchPbpMetaOptions()
      .then((meta) => {
        if (!alive) return;
        setOptions(meta);
        const season = meta.seasons?.[meta.seasons.length - 1] ?? "";
        const team = meta.teams?.includes("TOR") ? "TOR" : meta.teams?.[0] ?? "";
        setDraft((d) => ({ ...d, season: d.season || season, team: d.team || team }));
      })
      .catch(() => alive && setOptions({ seasons: [], teams: [], shotTypes: [], zones: [] }));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!applied) return;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setStatus("loading");
    fetchShotPage(applied, page, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setResult(data);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setResult(null);
        setStatus("error");
        setMessage(error instanceof ShotAccessError || error instanceof Error ? error.message : "Could not load shots.");
      });
    return () => controller.abort();
  }, [applied, page]);

  function search(event: FormEvent) {
    event.preventDefault();
    if (!canSearchShots(draft)) {
      setStatus("error");
      setMessage("Pick a season and team.");
      return;
    }
    setPage(1);
    setApplied({ ...draft });
  }

  async function exportFile(format: "csv" | "json") {
    if (!applied) return;
    setExporting(format);
    try {
      await authenticatedDownload(shotExportUrl(applied, format), shotExportFilename(applied, format));
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Export failed.");
    } finally {
      setExporting("");
    }
  }

  const rows = result?.rows ?? [];
  const columns = result?.columns ?? [];
  const totalPages = result ? Math.max(1, Math.ceil(result.total / SHOT_PAGE_SIZE)) : 1;
  const canExport = status === "ready" && !!result && result.total > 0 && !exporting;
  const set = (key: keyof ShotFilters) => (e: { target: { value: string } }) => setDraft({ ...draft, [key]: e.target.value });

  return (
    <section className={`card ${styles.panel}`} aria-label="Play-by-play shots">
      <div>
        <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>Play-by-play shots</h2>
        <p className="muted" style={{ fontSize: 13, margin: 0 }}>
          One row per shot with its court location, from the ESPN play-by-play data. Choose a season and team, then narrow it down.
        </p>
      </div>

      <form className={styles.filters} onSubmit={search} aria-label="Filter shots">
        <label className={styles.field}>
          Season
          <select className={styles.input} value={draft.season} onChange={set("season")}>
            <option value="">Choose…</option>
            {(options?.seasons ?? []).map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className={styles.field}>
          Team
          <select className={styles.input} value={draft.team} onChange={set("team")}>
            <option value="">Choose…</option>
            {(options?.teams ?? []).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className={styles.field}>
          Opponent
          <select className={styles.input} value={draft.opp} onChange={set("opp")}>
            <option value="">All</option>
            {(options?.teams ?? []).filter((t) => t !== draft.team).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className={styles.field}>
          Shot type
          <select className={styles.input} value={draft.shotType} onChange={set("shotType")}>
            <option value="">All</option>
            {(options?.shotTypes ?? []).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className={styles.field}>
          Zone
          <select className={styles.input} value={draft.zone} onChange={set("zone")}>
            <option value="">All</option>
            {(options?.zones ?? []).map((z) => <option key={z} value={z}>{z}</option>)}
          </select>
        </label>
        <label className={styles.field}>
          Result
          <select className={styles.input} value={draft.result} onChange={(e) => setDraft({ ...draft, result: e.target.value as ShotResult })}>
            <option value="">All</option>
            <option value="made">Made</option>
            <option value="missed">Missed</option>
          </select>
        </label>
        <div className={styles.actions}>
          <button className={`${styles.button} ${styles.primary}`} type="submit">Search</button>
        </div>
      </form>

      <div className={styles.bar}>
        <div className={styles.status} role="status" aria-live="polite">
          {status === "idle" ? "Pick a season and team, then Search." : null}
          {status === "loading" ? "Loading shots…" : null}
          {status === "ready" && result
            ? result.total === 0
              ? "No shots match these filters."
              : `${result.total.toLocaleString()} shot${result.total === 1 ? "" : "s"} • page ${result.page} of ${totalPages}`
            : null}
        </div>
        <div className={styles.actions}>
          <button className={styles.button} type="button" disabled={!canExport} onClick={() => void exportFile("csv")}>
            {exporting === "csv" ? "Exporting CSV…" : "Export CSV"}
          </button>
          <button className={styles.button} type="button" disabled={!canExport} onClick={() => void exportFile("json")}>
            {exporting === "json" ? "Exporting JSON…" : "Export JSON"}
          </button>
        </div>
      </div>

      {status === "error" ? <div className={styles.error} role="alert">{message}</div> : null}

      {status !== "error" && rows.length > 0 ? (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>{columns.map((c) => <th key={c.key}>{c.label}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={`${row.game_id}-${row.period}-${row.time_left}-${i}`}>
                  {columns.map((c) => (
                    <td key={c.key} className={c.key === "result" ? (row.result === "Made" ? styles.made : styles.missed) : undefined}>
                      {row[c.key] ?? "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className={styles.bar}>
        <button className={styles.button} type="button" disabled={page <= 1 || status === "loading"} onClick={() => setPage((p) => Math.max(1, p - 1))}>
          Previous
        </button>
        <p className={styles.note}>
          Exports use the searched filters, up to 10,000 shots{result?.dataset_version ? ` • data version ${result.dataset_version.slice(0, 19)}` : ""}.
        </p>
        <button className={styles.button} type="button" disabled={!result?.has_next || status === "loading"} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </section>
  );
}
