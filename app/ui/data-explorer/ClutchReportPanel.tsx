"use client";

// UI Layer — Data Explorer clutch report (SCRUM-629).
// Analysts pick a matchup and late-game situations, then compare both teams'
// shots, makes and points per shot by zone.

import React, { FormEvent, useEffect, useRef, useState } from "react";
import {
  clutchFilterProblem,
  DEFAULT_CLUTCH,
  describeClutch,
  type ClutchFilters,
  type ClutchReport,
} from "../../domain/clutch-report";
import { fetchClutchReport } from "../../services/clutch-report";

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

function toInt(value: string): number {
  return value.trim() === "" ? NaN : Number(value);
}

export default function ClutchReportPanel({ seasons, teams, teamNames, season = "", team = "" }: ClutchReportPanelProps) {
  const [draft, setDraft] = useState<ClutchFilters>({ season, team, opp: "", ...DEFAULT_CLUTCH });
  const [report, setReport] = useState<ClutchReport | null>(null);
  const [applied, setApplied] = useState<ClutchFilters | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const inFlight = useRef<AbortController | null>(null);

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
    </section>
  );
}
