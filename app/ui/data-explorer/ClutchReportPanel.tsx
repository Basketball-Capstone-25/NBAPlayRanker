"use client";

// UI Layer — Data Explorer clutch report (SCRUM-629).
// Analysts pick a matchup and late-game situations, then compare both teams'
// shots, makes and points per shot by zone.

import React, { FormEvent, useEffect, useRef, useState } from "react";
import { quickCallHref } from "../../domain/quick-call";
import {
  clutchFilterProblem,
  clutchQuickCallSituation,
  DEFAULT_CLUTCH,
  describeClutch,
  type ClutchFilters,
  type ClutchReport,
  type ZoneSplit,
} from "../../domain/clutch-report";
import {
  authenticatedDownload,
  clutchFilename,
  clutchReportUrl,
  fetchClutchReport,
  type ClutchFormat,
} from "../../services/clutch-report";

type Status = "idle" | "loading" | "ready" | "error";

export const TIME_OPTIONS = [30, 60, 120, 180, 300, 480, 720];

export type ClutchReportPanelProps = {
  seasons: readonly string[];
  teams: readonly string[];
  teamNames?: Record<string, string>;
  /** Starting season and team, usually the ones already picked in the Data Explorer. */
  season?: string;
  team?: string;
};

const SMALL_STYLE: React.CSSProperties = { background: "#fff4d6", color: "#7a4b00" };

function pps(split: ZoneSplit): string {
  return split.points_per_shot === null ? "—" : split.points_per_shot.toFixed(2);
}

function SplitCells({ split }: { split: ZoneSplit }) {
  const style = split.small_sample ? SMALL_STYLE : undefined;
  return (
    <>
      <td style={style} data-small-sample={split.small_sample ? "true" : undefined}>
        {split.shots}
        {split.small_sample ? <span title="Small sample"> *</span> : null}
      </td>
      <td style={style}>{split.makes}</td>
      <td style={style}>{pps(split)}</td>
    </>
  );
}

/** Both teams side by side by zone; zones under the small-sample threshold are shaded and starred. */
export function ClutchTable({ report }: { report: ClutchReport }) {
  const { team, opp } = report.filters;
  if (report.rows.length === 0) {
    return <p className="muted" style={{ fontSize: 13 }}>{report.message ?? "No shots match these situations."}</p>;
  }
  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table aria-label={`Clutch shots by zone, ${team} vs ${opp}`} style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "center" }}>
          <thead>
            <tr>
              <th />
              <th colSpan={3} scope="colgroup">{team}</th>
              <th colSpan={3} scope="colgroup">{opp}</th>
            </tr>
            <tr>
              <th scope="col" style={{ textAlign: "left" }}>Zone</th>
              {[team, opp].flatMap((t) => [
                <th key={`${t}-s`} scope="col">Shots</th>,
                <th key={`${t}-m`} scope="col">Makes</th>,
                <th key={`${t}-p`} scope="col">Pts/shot</th>,
              ])}
            </tr>
          </thead>
          <tbody>
            {report.rows.map((row) => (
              <tr key={row.zone}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 600 }}>{row.label}</th>
                <SplitCells split={row.team} />
                <SplitCells split={row.opp} />
              </tr>
            ))}
            <tr style={{ fontWeight: 700 }}>
              <th scope="row" style={{ textAlign: "left" }}>All zones</th>
              <SplitCells split={report.totals.team} />
              <SplitCells split={report.totals.opp} />
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12 }}>
        * Fewer than {report.small_sample_threshold} shots: a small sample, so treat its rate with caution.
        {report.dataset_version ? ` Data version ${report.dataset_version.slice(0, 19)}.` : ""}
      </p>
    </>
  );
}

export async function copyClutchCoachLink(filters: ClutchFilters, origin: string, writeText: (text: string) => Promise<void>) {
  const url = new URL(quickCallHref(clutchQuickCallSituation(filters)), origin).href;
  const reportKey = clutchReportUrl(filters);
  try {
    await writeText(url);
    return { url, reportKey, copied: true };
  } catch {
    return { url, reportKey, copied: false };
  }
}

type CoachLinkCopy = Awaited<ReturnType<typeof copyClutchCoachLink>>;

export function ClutchCoachLinkFeedback({ filters, copy }: { filters: ClutchFilters; copy: CoachLinkCopy | null }) {
  if (!copy || copy.reportKey !== clutchReportUrl(filters)) return null;
  return copy.copied ? <p role="status" className="muted" style={{ fontSize: 13 }}>Coach link copied.</p> :
    <label style={{ display: "grid", gap: 6, marginTop: 10, fontSize: 13 }}>
      Could not copy automatically. Select and copy this coach link:
      <input className="input" readOnly value={copy.url} onFocus={event => event.currentTarget.select()} />
    </label>;
}

/** Downloads and a Quick-Call link that the analyst can share with a coach. */
export function ClutchActions({ filters, downloading, onDownload, downloadError = "" }: {
  filters: ClutchFilters;
  downloading: "" | ClutchFormat;
  onDownload(format: ClutchFormat): void;
  /** A failed download is shown under the buttons and keeps the report on screen. */
  downloadError?: string;
}) {
  const [copy, setCopy] = useState<CoachLinkCopy | null>(null);
  async function copyLink() {
    setCopy(await copyClutchCoachLink(filters, window.location.origin, text => navigator.clipboard.writeText(text)));
  }
  return (
    <>
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
      <button className="btn btn--secondary" type="button" disabled={!!downloading} onClick={() => onDownload("pdf")}>
        {downloading === "pdf" ? "Preparing PDF…" : "Download PDF"}
      </button>
      <button className="btn btn--secondary" type="button" disabled={!!downloading} onClick={() => onDownload("csv")}>
        {downloading === "csv" ? "Preparing CSV…" : "Download CSV"}
      </button>
      <button className="btn" type="button" onClick={copyLink}>Copy coach link</button>
      <span className="muted" style={{ fontSize: 12 }}>For a coach account.</span>
    </div>
    {downloadError ? <p role="alert" style={{ fontSize: 13 }}>{downloadError}</p> : null}
    <ClutchCoachLinkFeedback filters={filters} copy={copy} />
    </>
  );
}

function toInt(value: string): number {
  return value.trim() === "" ? NaN : Number(value);
}

export default function ClutchReportPanel({ seasons, teams, teamNames, season = "", team = "" }: ClutchReportPanelProps) {
  const [draft, setDraft] = useState<ClutchFilters>({ season, team, opp: "", ...DEFAULT_CLUTCH });
  const [report, setReport] = useState<ClutchReport | null>(null);
  const [applied, setApplied] = useState<ClutchFilters | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const inFlight = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState<"" | ClutchFormat>("");

  // Follow the explorer's season and team until the analyst changes them here.
  useEffect(() => {
    setDraft((d) => ({ ...d, season: d.season || season, team: d.team || team }));
  }, [season, team]);
  useEffect(() => () => inFlight.current?.abort(), []);

  const set = <K extends keyof ClutchFilters>(key: K, value: ClutchFilters[K]) => setDraft((d) => ({ ...d, [key]: value }));
  const label = (abbr: string) => (teamNames?.[abbr] ? `${abbr} — ${teamNames[abbr]}` : abbr);

  async function run(event: FormEvent) {
    event.preventDefault();
    const problem = clutchFilterProblem(draft);
    if (problem) {
      setStatus("error");
      setMessage(problem);
      return;
    }
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    const filters = { ...draft };
    setStatus("loading");
    setDownloadError("");
    setMessage("");
    try {
      const data = await fetchClutchReport(filters, controller.signal);
      if (controller.signal.aborted) return;
      setReport(data);
      setApplied(filters);
      setStatus("ready");
    } catch (error) {
      if (controller.signal.aborted) return;
      setReport(null);
      setApplied(null);
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "The clutch report could not be built.");
    }
  }

  async function download(format: ClutchFormat) {
    if (!applied) return;
    setDownloading(format);
    setDownloadError("");
    try {
      // Uses the filters of the report on screen, not unsaved edits in the form.
      await authenticatedDownload(clutchReportUrl(applied, format), clutchFilename(applied, format));
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Download failed.");
    } finally {
      setDownloading("");
    }
  }

  return (
    <section className="card" aria-label="Clutch report" style={{ marginTop: 14 }}>
      <h2 style={{ margin: "6px 0 4px", fontSize: 16 }}>Clutch report</h2>
      <p className="muted" style={{ fontSize: 13, margin: 0 }}>
        Compare both teams&apos; late-game shots by zone from the play-by-play data. Margin is from each shooting team&apos;s side.
      </p>

      <form onSubmit={run} aria-label="Clutch situations" className="form-grid" style={{ marginTop: 10, gap: 10 }}>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Season
          <select className="input" value={draft.season} onChange={(e) => set("season", e.target.value)}>
            <option value="">Choose…</option>
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Your team
          <select className="input" value={draft.team} onChange={(e) => set("team", e.target.value)}>
            <option value="">Choose…</option>
            {teams.map((t) => <option key={t} value={t}>{label(t)}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Opponent
          <select className="input" value={draft.opp} onChange={(e) => set("opp", e.target.value)}>
            <option value="">Choose…</option>
            {teams.filter((t) => t !== draft.team).map((t) => <option key={t} value={t}>{label(t)}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          From period
          <select className="input" value={draft.periodFrom} onChange={(e) => set("periodFrom", Number(e.target.value))}>
            {[1, 2, 3, 4].map((p) => <option key={p} value={p}>{p === 4 ? "4th quarter + OT" : `Q${p} onward`}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Time left in period
          <select className="input" value={draft.secondsLeft} onChange={(e) => set("secondsLeft", Number(e.target.value))}>
            {TIME_OPTIONS.map((s) => <option key={s} value={s}>{s < 60 ? `${s} seconds or less` : `${s / 60} min or less`}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Margin from
          <input className="input" type="number" min={-40} max={40} value={Number.isNaN(draft.marginMin) ? "" : draft.marginMin}
            onChange={(e) => set("marginMin", toInt(e.target.value))} />
        </label>
        <label style={{ fontSize: 13, display: "grid", gap: 6 }}>
          Margin to
          <input className="input" type="number" min={-40} max={40} value={Number.isNaN(draft.marginMax) ? "" : draft.marginMax}
            onChange={(e) => set("marginMax", toInt(e.target.value))} />
        </label>
        <div style={{ display: "flex", alignItems: "end" }}>
          <button className="btn" type="submit" disabled={status === "loading"}>
            {status === "loading" ? "Building…" : "Build report"}
          </button>
        </div>
      </form>

      {status === "error" && message ? <p role="alert" style={{ fontSize: 13 }}>{message}</p> : null}
      {applied && report ? (
        <p className="muted" style={{ fontSize: 13 }}>
          {applied.team.toUpperCase()} vs {applied.opp.toUpperCase()}, {applied.season}: {describeClutch(applied)}.
        </p>
      ) : null}
      {status === "ready" && report ? <ClutchTable report={report} /> : null}
      {status === "ready" && applied ? <ClutchActions filters={applied} downloading={downloading} onDownload={download} downloadError={downloadError} /> : null}
    </section>
  );
}
