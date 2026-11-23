// Context + ML ranking page UI.

"use client";

import Link from "next/link";
import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { contextRank, fetchMetaOptions } from "../../../services/recommendation";
import { LatestRequest } from "../../../infrastructure/latest-request";
import ReportExport from "../../_shared/components/ReportExport";
import RankedPlayList from "./RankedPlayList";
import VoiceSituationButton from "./VoiceSituationButton";
import PlayerMadeShotsPanel from "./PlayerMadeShotsPanel";
import type { SituationField, VoiceSituation } from "../../../domain/voice-situation";
import { topPickIndexes } from "../../../domain/top-pick";
import { useAccountTeam } from "../../_shared/useAccountTeam";
import { contextReport, resultKey } from "../../../infrastructure/report-export";
import {
  clamp01,
  computeLateFactor,
  computeScoreFactors,
  categorizePlay,
  computeOverlay,
  contextReason,
  type ContextRow,
  type OverlayBreakdown,
} from "../../../domain/context-analysis";

type MetaOptions = {
  seasons: string[];
  teams: string[];
  teamNames?: Record<string, string>;
  playTypes?: string[];
  sides?: string[];
  hasMlPredictions?: boolean;
  _fallback?: boolean;
};

type DisplayRow = ContextRow & {
  finalPPP_display: number;
  deltaPPP_display: number;
  overlay: OverlayBreakdown;
};

function fmt(n: any, digits = 3) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

// SCRUM-824: nothing is pre-selected. A link may carry a matchup or situation
// (for example the clutch coach link); only values that exist in the data are used.
function linkedValue(value: string | null, allowed: string[]) {
  return value && allowed.includes(value) ? value : "";
}

function secondsToClock(sec: number) {
  const s = Math.max(0, Math.floor(Number(sec) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function Icon({
  name,
}: {
  name: "spark" | "play" | "clock" | "sliders" | "trend" | "copy";
}) {
  const common = {
    width: 16,
    height: 16,
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
  };
  switch (name) {
    case "spark":
      return (
        <svg {...common}>
          <path
            d="M12 2l1.2 4.2L17.4 7.4l-4.2 1.2L12 12.8l-1.2-4.2L6.6 7.4l4.2-1.2L12 2Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path
            d="M19 12l.7 2.4 2.3.6-2.3.6L19 18l-.7-2.4-2.3-.6 2.3-.6L19 12Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
        </svg>
      );
    case "play":
      return (
        <svg {...common}>
          <path d="M10 8.5v7l6-3.5-6-3.5Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" stroke="currentColor" strokeWidth="2" />
        </svg>
      );
    case "clock":
      return (
        <svg {...common}>
          <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z" stroke="currentColor" strokeWidth="2" />
          <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    case "sliders":
      return (
        <svg {...common}>
          <path d="M4 21v-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M4 10V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M12 21v-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M12 8V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M20 21v-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M20 12V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M2 14h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M10 12h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M18 16h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    case "trend":
      return (
        <svg {...common}>
          <path d="M3 17l6-6 4 4 8-8" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <path d="M17 7h4v4" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <path d="M9 9h10v10H9V9Z" stroke="currentColor" strokeWidth="2" />
          <path
            d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      );
  }
}

/** Shows results only while they match the inputs that produced them; otherwise a prompt to run. */
export function ContextResults({ current, loading, hasRows, matchupLabel, children }: {
  current: boolean;
  loading: boolean;
  hasRows: boolean;
  matchupLabel: string;
  children: React.ReactNode;
}) {
  return <div className="card resultsCard">
    <div className="sectionTitle">
      <h2>Top recommendations</h2>
      <span className="badge">{current && !loading ? matchupLabel : "Run to see results"}</span>
    </div>
    {!loading && !hasRows ? (
      <p className="muted" role="status">{current ? "No play-type data is available for this matchup and season. Try another season or team." : "Set the situation and click Run Context + ML."}</p>
    ) : loading ? (
      <p className="muted" role="status">Generating ML + context ranking…</p>
    ) : !current ? (
      <p className="muted" role="status">Inputs changed. Run Context + ML again.</p>
    ) : children}
  </div>;
}

// ✅ Inner component uses useSearchParams()
// ✅ Outer export wraps in Suspense
function ContextInner() {
  const params = useSearchParams();

  const [meta, setMeta] = useState<MetaOptions>({
    seasons: [],
    teams: [],
    teamNames: {},
    playTypes: [],
    sides: ["offense", "defense"],
    hasMlPredictions: false,
  });

  // Inputs
  const [season, setSeason] = useState("");
  const [our, setOur] = useState("");
  const [opp, setOpp] = useState("");
  const [k, setK] = useState(5);

  // Game situation: blank until the coach sets it (SCRUM-824).
  const [marginIn, setMarginIn] = useState<number | null>(null);
  const [periodIn, setPeriodIn] = useState<number | null>(null);
  const [timeIn, setTimeIn] = useState<number | null>(null);
  const situationSet = marginIn !== null && periodIn !== null && timeIn !== null;
  // SCRUM-626: fields a spoken phrase did not give are highlighted until set by hand.
  const [unclear, setUnclear] = useState<SituationField[]>([]);
  const needsHand = (field: SituationField, value: number | null) => value === null && unclear.includes(field);
  const handStyle = { outline: "2px solid #f59e0b", outlineOffset: 2 };
  // Only read once the situation is set; nothing is shown or sent before that.
  const margin = marginIn ?? 0;
  const period = periodIn ?? 1;
  const timeRemaining = timeIn ?? 720;

  const [applyOverlay, setApplyOverlay] = useState(true);
  const [overlayStrength, setOverlayStrength] = useState(1.0);

  const [liveMode, setLiveMode] = useState(false);

  // Details toggle (hidden by default, matches “finished app” feel)
  const [showBreakdown, setShowBreakdown] = useState(false);

  // Optional “Describe the situation” input (coach friendly)
  const [situationText, setSituationText] = useState("");

  // Results
  const [rows, setRows] = useState<ContextRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState("");
  const requestFilters = { season, our, opp, k, margin: marginIn, period: periodIn, time_remaining: timeIn };
  const currentKey = resultKey(requestFilters);
  const requestsRef = useRef(new LatestRequest());
  const lastAttemptedKeyRef = useRef("");

  const liveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const m = await fetchMetaOptions();
        if (cancelled) return;
        setMeta(m);

        const qsSeason = params.get("season") ?? "";
        const qsOur = params.get("our") ?? "";
        const qsOpp = params.get("opp") ?? "";
        const qsK = params.get("k") ?? "";

        const linkedSeason = linkedValue(qsSeason, m.seasons ?? []);
        const linkedOur = linkedValue(qsOur, m.teams ?? []);
        const linkedOpp = linkedValue(qsOpp, m.teams ?? []);

        const qsMargin = params.get("margin");
        const qsPeriod = params.get("period");
        // "time" is the name old Quick-Call links used (SCRUM-795).
        const qsTime = params.get("timeRemaining") ?? params.get("time");
        const qsOverlay = params.get("overlay");
        const qsStrength = params.get("strength");

        setSeason((prev) => prev || linkedSeason);
        setOur((prev) => prev || linkedOur);
        setOpp((prev) => prev || linkedOpp);

        if (qsK && Number.isFinite(Number(qsK))) setK(Math.min(10, Math.max(1, Number(qsK))));
        if (qsMargin && Number.isFinite(Number(qsMargin))) setMarginIn(Number(qsMargin));
        if (qsPeriod && Number.isFinite(Number(qsPeriod))) setPeriodIn(Number(qsPeriod));
        if (qsTime && Number.isFinite(Number(qsTime))) setTimeIn(Number(qsTime));
        if (qsOverlay === "0") setApplyOverlay(false);
        if (qsStrength && Number.isFinite(Number(qsStrength))) setOverlayStrength(Number(qsStrength));
      } catch (e: any) {
        console.error(e);
        if (!cancelled) setError(e?.message ?? "Failed to load metadata.");
      }
    }

    init();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // SCRUM-842: the account team is the only input pre-filled, and only when blank.
  const accountTeam = useAccountTeam(meta?.teams);
  useEffect(() => {
    if (accountTeam) setOur((prev) => prev || accountTeam);
  }, [accountTeam]);

  const metaReady = useMemo(
    () => (meta?.seasons?.length ?? 0) > 0 && (meta?.teams?.length ?? 0) > 0,
    [meta]
  );

  const canRun = useMemo(() => {
    if (!metaReady) return false;
    if (!season || !our || !opp) return false;
    if (our === opp) return false;
    return situationSet;
  }, [metaReady, season, our, opp, situationSet]);

  const baselineHref = useMemo(() => {
    const qs = new URLSearchParams({ season, our, opp, k: String(k) });
    return `/matchup?${qs.toString()}`;
  }, [season, our, opp, k]);

  const matchupChip = useMemo(() => {
    const ourName = meta.teamNames?.[our];
    const oppName = meta.teamNames?.[opp];
    const left = ourName ? `${our} (${ourName})` : our || "—";
    const right = oppName ? `${opp} (${oppName})` : opp || "—";
    return `${left} vs ${right}`;
  }, [meta.teamNames, our, opp]);

  function makeShareUrl() {
    const qs = new URLSearchParams({
      season,
      our,
      opp,
      k: String(k),
      ...(marginIn !== null ? { margin: String(marginIn) } : {}),
      ...(periodIn !== null ? { period: String(periodIn) } : {}),
      ...(timeIn !== null ? { timeRemaining: String(timeIn) } : {}),
      overlay: applyOverlay ? "1" : "0",
      strength: String(overlayStrength),
    });
    return `/context?${qs.toString()}`;
  }

  async function copyShareLink() {
    try {
      const url = makeShareUrl();
      await navigator.clipboard.writeText(url);
      setToast("Copied share link.");
      setTimeout(() => setToast(null), 1200);
    } catch {
      setToast("Could not copy. Copy the URL from the address bar.");
      setTimeout(() => setToast(null), 1800);
    }
  }

  async function runContext() {
    if (!season || !our || !opp) {
      setError("Please select a season, our team, and an opponent.");
      return;
    }
    if (!situationSet) {
      setError("Set the score margin, period and time remaining.");
      return;
    }
    if (our === opp) {
      setError("Our team and opponent must be different.");
      return;
    }

    lastAttemptedKeyRef.current = currentKey;
    await requestsRef.current.run(currentKey, () => contextRank({
      season, our, opp, margin, period, timeRemaining, k,
    }), {
      start: () => {
        setLoading(true);
        setError(null);
        setRows([]);
        setLoadedKey("");
      },
      success: (out) => {
        setRows(Array.isArray(out) ? out : []);
        setLoadedKey(currentKey);
      },
      error: (reason) => {
        setError(reason instanceof Error ? reason.message : "Failed to generate context+ML recommendations.");
        setRows([]);
      },
      settled: () => setLoading(false),
    }, true);
  }


  useEffect(() => {
    if (!liveMode) return;
    if (!canRun) return;
    if (loading || lastAttemptedKeyRef.current === currentKey) return;

    if (liveTimerRef.current) clearTimeout(liveTimerRef.current);
    liveTimerRef.current = setTimeout(() => {
      runContext();
    }, 450);

    return () => {
      if (liveTimerRef.current) clearTimeout(liveTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveMode, currentKey, canRun, loading]);

  const meanMlPPP = useMemo(() => {
    const vals = rows.map((r) => Number(r.mlPPP)).filter((x) => Number.isFinite(x));
    if (!vals.length) return 0;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [rows]);

  const lateFactorNow = useMemo(() => computeLateFactor(period, timeRemaining), [period, timeRemaining]);
  const scoreFactorsNow = useMemo(() => computeScoreFactors(margin), [margin]);

  const displayRows: DisplayRow[] = useMemo(() => {
    if (!rows.length) return [];

    const mapped: DisplayRow[] = rows.map((r) => {
      const overlay = applyOverlay
        ? computeOverlay(r, meanMlPPP, period, timeRemaining, margin, overlayStrength)
        : {
            overlayPPP: 0,
            overlayEff: 0,
            overlayQuick: 0,
            overlayProtect: 0,
            overlayType: 0,
            lateFactor: computeLateFactor(period, timeRemaining),
            trailingFactor: computeScoreFactors(margin).trailing,
            leadingFactor: computeScoreFactors(margin).leading,
            varianceProxy: Math.abs(Number(r.mlPPP) - Number(r.baselinePPP)),
          };

      const finalPPP_display = Number(r.finalPPP) + Number(overlay.overlayPPP);
      const deltaPPP_display = finalPPP_display - Number(r.baselinePPP);

      return { ...r, finalPPP_display, deltaPPP_display, overlay };
    });

    const sorted = [...mapped].sort((a, b) => b.finalPPP_display - a.finalPPP_display);
    return sorted.slice(0, Math.min(10, Math.max(1, k)));
  }, [rows, applyOverlay, meanMlPPP, period, timeRemaining, margin, overlayStrength, k]);

  const resultsCurrent = loadedKey === currentKey && displayRows.length > 0 && !loading && !error;
  const exportPayload = resultsCurrent ? contextReport(
    { ...requestFilters, overlay: applyOverlay, overlay_strength: overlayStrength }, displayRows,
  ) : null;

  // SCRUM-496: the #1 play and any play tied with it on the displayed score.
  const topPicks = useMemo(() => new Set(topPickIndexes(displayRows.map((r) => r.finalPPP_display))), [displayRows]);
  const topTied = topPicks.size > 1;

  // SCRUM-626: a spoken situation fills what was understood and clears the rest,
  // so an old value never mixes with a new phrase. The coach still clicks Run.
  function applyVoice(result: VoiceSituation, heard: string) {
    setMarginIn(result.margin);
    setPeriodIn(result.period);
    setTimeIn(result.timeRemaining);
    setUnclear(result.unclear);
    setSituationText(heard);
  }

  // Quick preset buttons.
  function applyPreset(p: { margin: number; period: number; time: number; text: string }) {
    setMarginIn(p.margin);
    setPeriodIn(p.period);
    setTimeIn(p.time);
    setSituationText(p.text);
  }

  return (
    <section className="pageRoot">
      <style jsx>{`
        .pageRoot {
          max-width: 1200px;
          margin: 0 auto;
          padding: 28px 20px 48px;
          color: #17243d;
        }

        .hero {
          display: grid;
          gap: 10px;
          margin-bottom: 22px;
        }

        .heroTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(15, 23, 42, 0.03);
          color: rgba(15, 23, 42, 0.75);
          font-size: 12px;
          white-space: nowrap;
        }

        h1 {
          margin: 0;
          font-size: clamp(26px, 3vw, 34px);
          letter-spacing: -0.035em;
          line-height: 1.15;
        }

        .sub {
          margin: 0;
          color: rgba(15, 23, 42, 0.7);
          font-size: 13px;
          line-height: 1.45;
        }

        .card {
          border-radius: 18px;
          border: 1px solid #e3e7ef;
          background: #fff;
          padding: 22px;
          box-shadow: 0 4px 18px rgba(23, 36, 61, 0.025);
          overflow: hidden;
        }

        .grid2 {
          display: grid;
          grid-template-columns: 1fr;
          gap: 12px;
        }
        @media (min-width: 860px) {
          .grid2 {
            grid-template-columns: 1.05fr 0.95fr;
            align-items: start;
          }
        }

        .sectionTitle {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }
        .sectionTitle h2 {
          margin: 0;
          font-size: 17px;
          letter-spacing: -0.02em;
        }

        .inputs {
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
        }
        @media (min-width: 640px) {
          .inputs {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }
        }

        label {
          display: grid;
          gap: 6px;
          font-size: 12px;
          color: rgba(15, 23, 42, 0.8);
        }

        .input {
          width: 100%;
          border-radius: 12px;
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: rgba(255, 255, 255, 0.95);
          padding: 10px 10px;
          font-size: 13px;
          color: rgba(15, 23, 42, 0.92);
          outline: none;
        }

        textarea.input {
          min-height: 92px;
          resize: vertical;
        }

        .presetRow {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 10px;
        }

        .chipBtn {
          border-radius: 999px;
          padding: 8px 10px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(255, 255, 255, 0.65);
          cursor: pointer;
          font-size: 12px;
          color: rgba(15, 23, 42, 0.8);
        }
        .chipBtn:hover {
          background: rgba(255, 255, 255, 0.85);
        }

        .row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(15, 23, 42, 0.08);
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          border-radius: 10px;
          border: 1px solid #d8deea;
          background: rgba(255, 255, 255, 0.9);
          color: rgba(15, 23, 42, 0.92);
          font-weight: 650;
          font-size: 13px;
          text-decoration: none;
          cursor: pointer;
        }

        .btnPrimary {
          border: 1px solid #6950d4;
          background: #6950d4;
          color: #fff;
          box-shadow: 0 3px 8px rgba(105, 80, 212, 0.14);
        }

        .btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }

        .muted {
          color: rgba(15, 23, 42, 0.68);
          font-size: 12px;
          margin: 0;
        }

        .error {
          margin-top: 10px;
          font-size: 12px;
          color: rgba(185, 28, 28, 0.92);
        }

        .toast {
          margin-top: 10px;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(15, 23, 42, 0.03);
          padding: 10px;
          font-size: 12px;
          color: rgba(15, 23, 42, 0.8);
        }

        .ctxGrid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
        }
        @media (min-width: 760px) {
          .ctxGrid {
            grid-template-columns: 1fr 1fr;
          }
        }

        .ctxBox {
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(15, 23, 42, 0.02);
          padding: 10px;
          display: grid;
          gap: 8px;
        }

        .ctxTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          flex-wrap: wrap;
        }

        .range {
          width: 100%;
        }

        .meter {
          height: 10px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.1);
          overflow: hidden;
        }

        .meter > div {
          height: 100%;
          background: #7964cf;
        }

        .mono {
          font-family: var(
            --mono,
            ui-monospace,
            SFMono-Regular,
            Menlo,
            Monaco,
            Consolas,
            "Liberation Mono",
            "Courier New",
            monospace
          );
          font-size: 12px;
        }

        .resultsCard {
          margin-top: 12px;
        }






        details.moreInfo {
          margin-top: 14px;
          border-top: 1px solid rgba(15, 23, 42, 0.08);
          padding-top: 12px;
        }
        details.moreInfo summary {
          cursor: pointer;
          font-weight: 900;
          color: rgba(15, 23, 42, 0.9);
          list-style: disclosure-closed;
        }


        .tableWrap {
          overflow-x: auto;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(255, 255, 255, 0.86);
          margin-top: 10px;
        }

        table {
          width: max-content;
          min-width: 100%;
          border-collapse: collapse;
        }

        th,
        td {
          padding: 10px 10px;
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          text-align: left;
          font-size: 12px;
          white-space: nowrap;
        }

        th {
          font-weight: 900;
          color: rgba(15, 23, 42, 0.82);
          background: rgba(15, 23, 42, 0.02);
        }
        .advanced {
          margin-top: 18px;
          border-top: 1px solid #e9ecf2;
          padding-top: 14px;
        }
        .advanced > summary {
          cursor: pointer;
          font-size: 13px;
          font-weight: 650;
          color: #55439f;
          padding: 4px 0;
        }
        .advanced[open] > summary { margin-bottom: 14px; }
        .input:focus-visible, .btn:focus-visible, .chipBtn:focus-visible,
        .range:focus-visible, summary:focus-visible {
          outline: 3px solid #b8a9ef;
          outline-offset: 3px;
        }
        .btn:not(:disabled):hover { border-color: #a99ad9; background: #f7f5fd; }
        .btnPrimary:not(:disabled):hover { background: #5940bd; color: #fff; }
        .range { accent-color: #6950d4; }
        .emptyState {
          margin: 0;
          padding: 28px 18px;
          border-radius: 12px;
          background: #f7f8fc;
          color: #59677c;
          text-align: center;
          font-size: 14px;
          line-height: 1.6;
        }
        @media (max-width: 639px) {
          .pageRoot { padding: 20px 12px 36px; }
          .card { padding: 16px; }
          .badge { white-space: normal; }
          .heroTop { align-items: flex-start; }
        }
      `}</style>

      {/* Header */}
      <div className="hero">
        <div className="heroTop">
          <div className="badge">
            <Icon name="spark" /> In-game planning
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href={baselineHref}>
              Compare Baseline
            </Link>
            <Link className="btn" href="/gameplan">
              Gameplan
            </Link>
          </div>
        </div>

        <h1>Find the right play for this moment</h1>
        <p className="sub">Choose a matchup, add the score and clock, then compare your options.</p>

        <div className="presetRow">
          <button
            className="chipBtn"
            type="button"
            onClick={() =>
              applyPreset({
                margin: -2,
                period: 4,
                time: 180,
                text: "Down 2 with 3:00 left in the 4th. We need a good look. After timeout.",
              })
            }
          >
            Down 2 • 3:00 • Q4
          </button>
          <button
            className="chipBtn"
            type="button"
            onClick={() =>
              applyPreset({
                margin: -6,
                period: 4,
                time: 120,
                text: "Down 6 with 2:00 left in the 4th. Need quick scores.",
              })
            }
          >
            Down 6 • 2:00 • Q4
          </button>
          <button
            className="chipBtn"
            type="button"
            onClick={() =>
              applyPreset({
                margin: +4,
                period: 4,
                time: 90,
                text: "Up 4 with 1:30 left in the 4th. Protect the lead and get a solid look.",
              })
            }
          >
            Up 4 • 1:30 • Q4
          </button>
          <button
            className="chipBtn"
            type="button"
            onClick={() =>
              applyPreset({
                margin: 0,
                period: 5,
                time: 120,
                text: "Tie game in OT. Need a reliable shot and good rebounding positions.",
              })
            }
          >
            Tie • 2:00 • OT
          </button>
        </div>
      </div>

      {/* Inputs */}
      <div className="card">
        <div className="sectionTitle">
          <h2>Choose your matchup</h2>
          <span className="badge">{metaReady ? "Ready" : meta?._fallback ? "Data unavailable" : "Loading…"}</span>
        </div>

        <div className="inputs">
          <label>
            Season
            <select className="input" value={season} onChange={(e) => setSeason(e.target.value)}>
              <option value="" disabled>Choose a season</option>
              {(meta.seasons ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label>
            Plays to show <span className="muted" style={{ fontWeight: 400 }}>(model setting)</span>
            <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[3, 5, 7, 10].map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
          </label>

          <label>
            Our Team
            <select className="input" value={our} onChange={(e) => setOur(e.target.value)}>
              <option value="" disabled>Choose our team</option>
              {(meta.teams ?? []).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label>
            Opponent
            <select className="input" value={opp} onChange={(e) => setOpp(e.target.value)}>
              <option value="" disabled>Choose an opponent</option>
              {(meta.teams ?? []).map((t) => (
                <option key={t} value={t} disabled={t === our}>
                  {t}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div style={{ marginTop: 12 }} data-testid="voice-situation">
          <VoiceSituationButton onResult={applyVoice} />
        </div>

        <div className="ctxGrid" style={{ marginTop: 12 }}>
          <div className="ctxBox">
            <div className="ctxTop">
              <div>
                <div style={{ fontWeight: 700, fontSize: 13 }}>Score margin</div>
                <p className="muted">Our score minus theirs. Negative means we’re trailing.</p>
              </div>
              <input
                className="input"
                type="number"
                step={1}
                placeholder="e.g. -3"
                style={needsHand("margin", marginIn) ? { width: 110, ...handStyle } : { width: 110 }}
                aria-label="Score margin"
                value={marginIn ?? ""}
                onChange={(e) => setMarginIn(e.target.value === "" ? null : Number(e.target.value))}
              />
            </div>

            <input
              className="range"
              type="range"
              aria-label="Score margin slider"
              min={-15}
              max={15}
              step={1}
              value={marginIn ?? 0}
              style={{ opacity: marginIn === null ? 0.35 : 1 }}
              onChange={(e) => setMarginIn(Number(e.target.value))}
            />

            <p className="muted">
              {marginIn === null ? (
                "Not set: type the margin or drag the slider."
              ) : (
                <>
                  trailing <strong>{fmt(scoreFactorsNow.trailing, 2)}</strong> • leading{" "}
                  <strong>{fmt(scoreFactorsNow.leading, 2)}</strong>
                </>
              )}
            </p>
          </div>

          <div className="ctxBox">
            <div className="ctxTop">
              <div>
                <div style={{ fontWeight: 900, fontSize: 12 }}>Time pressure</div>
                <p className="muted">Set period + time remaining</p>
              </div>

              <select className="input" aria-label="Period" value={periodIn ?? ""} onChange={(e) => setPeriodIn(Number(e.target.value))} style={needsHand("period", periodIn) ? { width: 110, ...handStyle } : { width: 110 }}>
                <option value="" disabled>Period</option>
                <option value={1}>P1</option>
                <option value={2}>P2</option>
                <option value={3}>P3</option>
                <option value={4}>P4</option>
                <option value={5}>OT</option>
              </select>
            </div>

            <input
              className="range"
              type="range"
              min={0}
              max={720}
              step={10}
              aria-label="Time remaining in the period"
              value={timeIn ?? 360}
              style={needsHand("timeRemaining", timeIn) ? { opacity: 0.6, ...handStyle } : { opacity: timeIn === null ? 0.35 : 1 }}
              onChange={(e) => setTimeIn(Number(e.target.value))}
            />

            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
              <div className="muted">
                <Icon name="clock" /> <strong>{timeIn === null ? "Not set: drag to the time left" : secondsToClock(timeRemaining)}</strong>
              </div>
              <div className="muted">
                late <strong>{periodIn === null || timeIn === null ? "—" : fmt(lateFactorNow, 2)}</strong>
              </div>
            </div>

            <div className="meter" aria-label="late factor meter">
              <div style={{ width: `${periodIn === null || timeIn === null ? 0 : clamp01(lateFactorNow) * 100}%` }} />
            </div>
          </div>
        </div>

        <div className="ctxGrid" style={{ marginTop: 12 }}>
          <div className="ctxBox">
            <div className="ctxTop">
              <div>
                <div style={{ fontWeight: 900, fontSize: 12 }}>Describe the situation</div>
                <p className="muted">Optional notes to keep with this scenario.</p>
              </div>
            </div>
            <textarea className="input" aria-label="Situation notes" placeholder="Optional notes, for example: need a good look after the timeout." value={situationText} onChange={(e) => setSituationText(e.target.value)} />
          </div>

          <details className="advanced">
            <summary>Ranking adjustments & live updates</summary>
            <div className="ctxTop">
              <div>
                <div style={{ fontWeight: 900, fontSize: 12 }}>Context adjustment</div>
                <p className="muted">Small rule-based adjustments, applied after the model.</p>
              </div>
              <span className="badge">
                <Icon name="sliders" /> strength {fmt(overlayStrength, 1)}
              </span>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <label className="muted" style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
                <input type="checkbox" checked={applyOverlay} onChange={(e) => setApplyOverlay(e.target.checked)} />
                Apply overlay
              </label>

              <label className="muted" style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
                Strength
                <input
                  className="input"
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={overlayStrength}
                  onChange={(e) => setOverlayStrength(Number(e.target.value))}
                  style={{ width: 110 }}
                />
              </label>

              <label className="muted" style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
                <input type="checkbox" checked={liveMode} onChange={(e) => setLiveMode(e.target.checked)} />
                Live mode
              </label>
            </div>

          </details>
        </div>

        <div className="row" style={{ marginTop: 10, paddingTop: 10 }}>
          <button className="btn btnPrimary" type="button" onClick={runContext} disabled={!canRun || loading}>
            <Icon name="play" /> {loading ? "Running…" : "Run Context + ML"}
          </button>

          <ReportExport payload={exportPayload}
            disabledReason={loading ? "Wait for the rankings to finish." : !resultsCurrent && rows.length ? "Inputs changed. Run Context + ML again before exporting." : undefined}
            description="Export these rankings, including the selected context overlay." />

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn" type="button" onClick={copyShareLink}>
              <Icon name="copy" /> Copy share link
            </button>
            <button className="btn" type="button" onClick={() => setShowBreakdown((v) => !v)}>
              <Icon name="trend" /> {showBreakdown ? "Hide breakdown" : "Show breakdown"}
            </button>
          </div>
        </div>

        {error ? <div className="error" role="alert">{error}</div> : null}
        {toast ? <div className="toast" role="status">{toast}</div> : null}

        <div className="row">
          <span className="badge">
            {situationSet
              ? `${matchupChip} • P${period} • ${secondsToClock(timeRemaining)} • margin ${margin >= 0 ? `+${margin}` : margin}`
              : "Choose the matchup and the game situation, then run."}
          </span>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href={baselineHref}>
              Compare Baseline
            </Link>
          </div>
        </div>
      </div>

      {/* Results */}
      <ContextResults current={loadedKey === currentKey && !loading && !error} loading={loading} hasRows={displayRows.length > 0} matchupLabel={`${our} vs ${opp} • ${season}`}>
            {/* SCRUM-795: one large list, same style the Quick-Call screen used. */}
            <RankedPlayList
              plays={displayRows.map((r) => ({
                playType: r.playType,
                ppp: Number(r.finalPPP_display),
                shotZone: r.shotZone,
                reason: contextReason(Number(r.deltaPPP_display), r.contextLabel),
              }))}
              current={resultsCurrent}
              topPicks={topPicks}
              tied={topTied}
            />
            {topTied ? (
              <p className="muted" style={{ fontSize: 12, marginTop: 8 }} data-testid="top-pick-tie">
                #1 is tied with {[...topPicks].slice(1).map((i) => displayRows[i].playType).join(", ")} on PPP
              </p>
            ) : null}

            {/* Optional breakdown table */}
            {showBreakdown ? (
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Play Type</th>
                      <th>PPP_display</th>
                      <th>Backend Final</th>
                      <th>ML</th>
                      <th>Baseline</th>
                      <th>Δ_display</th>
                      <th>Overlay</th>
                      <th>Eff</th>
                      <th>Quick</th>
                      <th>Protect</th>
                      <th>Type</th>
                      <th>Late</th>
                      <th>Trail</th>
                      <th>Lead</th>
                      <th>Var</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayRows.map((r, idx) => {
                      const d = Number(r.deltaPPP_display);
                      const sign = d >= 0 ? "+" : "";
                      return (
                        <tr key={`${r.playType}-${idx}`} className={topPicks.has(idx) ? "topPickRow" : undefined}>
                          <td>{idx + 1}</td>
                          <td style={{ fontWeight: 900 }}>{r.playType}</td>
                          <td>{fmt(r.finalPPP_display, 3)}</td>
                          <td>{fmt(r.finalPPP, 3)}</td>
                          <td>{fmt(r.mlPPP, 3)}</td>
                          <td>{fmt(r.baselinePPP, 3)}</td>
                          <td>
                            {sign}
                            {fmt(d, 3)}
                          </td>
                          <td>{fmt(r.overlay.overlayPPP, 3)}</td>
                          <td>{fmt(r.overlay.overlayEff, 3)}</td>
                          <td>{fmt(r.overlay.overlayQuick, 3)}</td>
                          <td>{fmt(r.overlay.overlayProtect, 3)}</td>
                          <td>{fmt(r.overlay.overlayType, 3)}</td>
                          <td>{fmt(r.overlay.lateFactor, 2)}</td>
                          <td>{fmt(r.overlay.trailingFactor, 2)}</td>
                          <td>{fmt(r.overlay.leadingFactor, 2)}</td>
                          <td>{fmt(r.overlay.varianceProxy, 3)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
      </ContextResults>

      {/* SCRUM-825: where a chosen player made shots against this opponent. */}
      <PlayerMadeShotsPanel our={our} opp={opp} />

      {/* ✅ More info (all explanations go here, at the bottom) */}
      <details className="moreInfo">
        <summary>How these rankings work</summary>

        <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
          <div className="card" style={{ padding: 12 }}>
            <div className="sectionTitle" style={{ marginBottom: 6 }}>
              <h2>What this page does</h2>
              <span className="badge">Explainability</span>
            </div>
            <p className="muted" style={{ lineHeight: 1.6 }}>
              The backend returns <span className="mono">finalPPP</span> (context-aware result) plus{" "}
              <span className="mono">mlPPP</span> and <span className="mono">baselinePPP</span>. This page optionally adds a small,
              transparent overlay (<span className="mono">overlayPPP</span>) based on margin/time to re-rank Top-K.
            </p>
          </div>

          <div className="card" style={{ padding: 12 }}>
            <div className="sectionTitle" style={{ marginBottom: 6 }}>
              <h2>Overlay inputs (how they affect ranking)</h2>
              <span className="badge">Policy</span>
            </div>
            <ul className="muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.65 }}>
              <li>
                <strong>Late factor</strong> ramps in the last ~3 minutes and changes by period (Q4/OT are most sensitive).
              </li>
              <li>
                <strong>Trailing</strong> increases emphasis on quick/efficient looks; <strong>Leading</strong> reduces variance/risk.
              </li>
              <li>
                <strong>Strength</strong> is clamped (0–2) so overlay can’t dominate the backend model.
              </li>
            </ul>
          </div>

          <div className="card" style={{ padding: 12 }}>
            <div className="sectionTitle" style={{ marginBottom: 6 }}>
              <h2>Share link</h2>
              <span className="badge">Reproducible</span>
            </div>
            <p className="muted" style={{ lineHeight: 1.6 }}>
              Copy share link saves season/teams/k + margin/period/time + overlay settings in the URL so the scenario is reproducible.
            </p>
          </div>

        </div>
      </details>
    </section>
  );
}

/** Context ML page wrapped in Suspense so share-link query parameters can be read. */
export default function ContextPage() {
  return (
    <Suspense
      fallback={
        <section className="card">
          <div style={{ padding: 16 }}>Loading…</div>
        </section>
      }
    >
      <ContextInner />
    </Suspense>
  );
}
