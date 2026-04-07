// Gameplan page UI.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Gameplan.module.css";

import {
  baselineRank,
  contextRank,
  fetchMetaOptions,
  fetchPlaytypeViz,
  postJson,
  type MetaOptions,
} from "../../../services/recommendation";
import {
  fetchShotHeatmap,
  fetchShotPlanRank,
  getShotPlanPdfUrl,
  authenticatedDownload,
} from "../../../services/shotAnalysis";
import { gradeFromRank, coachingCuesFor, counterPlanFor } from "../../../domain/coaching-knowledge";
import {
  normalizeTeams,
  normalizeWhy,
  normalizePlay,
  type NormalizedPlay,
  type TeamOption,
} from "../../../services/gameplan";

type PlanItem = { id: string; label: string };

type RoleAssignments = {
  ballHandler: string;
  screener: string;
  cornerSpacer: string;
  cutter: string;
  safety: string;
};

type NlpContext = {
  period?: number | null;
  time_remaining?: number | null; // returned by NLP service (snake), we map it internally
  margin?: number | null;
  after_timeout?: boolean | null;
  late_clock?: boolean | null;
  need3?: boolean | null;
  protect_lead?: boolean | null;
  end_of_quarter?: boolean | null;
  vs_switching?: boolean | null;
  defense_style?: string | null;
  pace?: string | null;
  must_stop?: boolean | null;
  quick2?: boolean | null;
  two_for_one?: boolean | null;
};

type NlpParseResponse = {
  context: NlpContext;
  confidence: number;
  clarifying_questions?: string[];
  warnings?: string[];
};

type ExplainItem = {
  play_type?: string;
  summary?: string;
  evidence?: string[];
  caution?: string;
};

type NlpExplainResponse = {
  mode: "baseline" | "context-ml";
  explanation: ExplainItem[];
};

const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE as string | undefined) ||
  "http://127.0.0.1:8000";

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function prettyTime(seconds: number) {
  const s = clamp(seconds, 0, 720);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
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

function Chip({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`${styles.chip} ${active ? styles.chipActive : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      {label}
    </button>
  );
}

export default function GameplanClient() {
  const [options, setOptions] = useState<MetaOptions | null>(null);
  const [optError, setOptError] = useState<string | null>(null);

  // Situation
  const [season, setSeason] = useState("2024-25");
  const [our, setOur] = useState("TOR");
  const [opp, setOpp] = useState("BOS");

  // Friendly focus slider (0..100)
  const [focus, setFocus] = useState(35);

  // Context
  const [quarter, setQuarter] = useState(4);
  const [timeLeft, setTimeLeft] = useState(180);
  const [margin, setMargin] = useState(-2);

  const [ctxAfterTimeout, setCtxAfterTimeout] = useState(false);
  const [ctxLateClock, setCtxLateClock] = useState(true);
  const [ctxNeed3, setCtxNeed3] = useState(false);
  const [ctxProtectLead, setCtxProtectLead] = useState(false);
  const [ctxEndQ, setCtxEndQ] = useState(false);
  const [ctxVsSwitching, setCtxVsSwitching] = useState(false);

  // Natural language (NLP)
  const [nlText, setNlText] = useState<string>(
    "Down 2 with 3:00 left in the 4th. We need a good look. After timeout."
  );
  const [nlConfidence, setNlConfidence] = useState<number | null>(null);
  const [nlQuestions, setNlQuestions] = useState<string[]>([]);
  const [nlWarnings, setNlWarnings] = useState<string[]>([]);

  // Controls
  const [showPlays, setShowPlays] = useState(5);
  const [recommendationStyle, setRecommendationStyle] = useState<"smart" | "baseline">("smart");

  // Results
  const [loading, setLoading] = useState(false);
  const [baselinePlays, setBaselinePlays] = useState<NormalizedPlay[]>([]);
  const [smartPlays, setSmartPlays] = useState<NormalizedPlay[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // NLP explanations for top plays
  const [explainMap, setExplainMap] = useState<Record<string, ExplainItem>>({});

  // Visuals
  const [vizTab, setVizTab] = useState<"playZones" | "shotHeatmap">("playZones");
  const [playVizLoading, setPlayVizLoading] = useState(false);
  const [playVizError, setPlayVizError] = useState<string | null>(null);
  const [playVizCaption, setPlayVizCaption] = useState<string>("");
  const [playVizBase64, setPlayVizBase64] = useState<string>("");

  const [shotVizLoading, setShotVizLoading] = useState(false);
  const [shotVizError, setShotVizError] = useState<string | null>(null);
  const [shotVizCaption, setShotVizCaption] = useState<string>("");
  const [shotVizBase64, setShotVizBase64] = useState<string>("");

  // Shot plan (Dataset2)
  const [shotLoading, setShotLoading] = useState(false);
  const [shotError, setShotError] = useState<string | null>(null);
  const [topShotTypes, setTopShotTypes] = useState<any[]>([]);
  const [topZones, setTopZones] = useState<any[]>([]);
  const [selectedShotType, setSelectedShotType] = useState<string>("");
  const [selectedZone, setSelectedZone] = useState<string>("");

  // Plan / notes / roles
  const [plan, setPlan] = useState<PlanItem[]>([]);
  const [notesByPlay, setNotesByPlay] = useState<Record<string, string>>({});
  const [roles, setRoles] = useState<RoleAssignments>({
    ballHandler: "",
    screener: "",
    cornerSpacer: "",
    cutter: "",
    safety: "",
  });

  const [statusHint, setStatusHint] = useState<string>("");

  // Used to ignore stale async responses when the user rebuilds quickly.
  const latestBuildRef = useRef<number>(0);

  // Load options
  useEffect(() => {
    let alive = true;
    (async () => {
      setOptError(null);
      try {
        const data = await fetchMetaOptions();
        if (!alive) return;
        setOptions(data);

        const seasons = data?.seasons ?? ["2024-25"];
        const teams = normalizeTeams(data);

        setSeason((prev) => (seasons.includes(prev) ? prev : seasons[0]));
        const codes = new Set(teams.map((t) => t.code));
        setOur((prev) => (codes.has(prev) ? prev : teams[0]?.code ?? "TOR"));
        setOpp((prev) => {
          if (codes.has(prev) && prev !== (teams[0]?.code ?? "TOR")) return prev;
          const fallback = teams.find((t) => t.code !== (teams[0]?.code ?? "TOR"));
          return fallback?.code ?? "BOS";
        });
      } catch (e: any) {
        if (!alive) return;
        setOptError(e?.message || "Couldn’t load team/season options.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Load saved plan/notes/roles
  useEffect(() => {
    const rawNotes = safeLocalGet("nbaPlayRanker_gameplan_notes_v2");
    if (rawNotes) {
      try {
        setNotesByPlay(JSON.parse(rawNotes));
      } catch {}
    }

    const rawPlan = safeLocalGet("nbaPlayRanker_gameplan_plan_v2");
    if (rawPlan) {
      try {
        setPlan(JSON.parse(rawPlan));
      } catch {}
    }

    const rawRoles = safeLocalGet("nbaPlayRanker_gameplan_roles_v2");
    if (rawRoles) {
      try {
        setRoles(JSON.parse(rawRoles));
      } catch {}
    }
  }, []);

  useEffect(() => {
    safeLocalSet("nbaPlayRanker_gameplan_notes_v2", JSON.stringify(notesByPlay));
  }, [notesByPlay]);

  useEffect(() => {
    safeLocalSet("nbaPlayRanker_gameplan_plan_v2", JSON.stringify(plan));
  }, [plan]);

  useEffect(() => {
    safeLocalSet("nbaPlayRanker_gameplan_roles_v2", JSON.stringify(roles));
  }, [roles]);

  // Derived weights
  const wDef = useMemo(() => clamp(focus / 100, 0, 1), [focus]);
  const wOff = useMemo(() => Number((1 - wDef).toFixed(2)), [wDef]);

  const teamOptions = useMemo(() => normalizeTeams(options), [options]);
  const seasonOptions = useMemo(() => options?.seasons ?? ["2024-25"], [options]);

  const activePlays = useMemo(() => {
    if (recommendationStyle === "baseline") return baselinePlays;
    return smartPlays.length ? smartPlays : baselinePlays;
  }, [recommendationStyle, baselinePlays, smartPlays]);

  const selectedPlay = useMemo(() => {
    if (!activePlays.length) return null;
    if (!selectedId) return activePlays[0];
    return activePlays.find((p) => p.id === selectedId) ?? activePlays[0];
  }, [activePlays, selectedId]);

  const top3 = useMemo(() => activePlays.slice(0, 3), [activePlays]);

  const situationSummary = useMemo(() => {
    const upDown =
      margin === 0 ? "Tied" : margin > 0 ? `Up ${margin}` : `Down ${Math.abs(margin)}`;
    const t = prettyTime(timeLeft);
    return `Q${quarter} • ${t} left • ${upDown}`;
  }, [quarter, timeLeft, margin]);

  function addToPlan(play: NormalizedPlay) {
    setPlan((prev) => {
      if (prev.some((p) => p.id === play.id)) return prev;
      return [...prev, { id: play.id, label: play.playType }];
    });
    setStatusHint("Added to your plan ✅");
    setTimeout(() => setStatusHint(""), 900);
  }

  function removeFromPlan(id: string) {
    setPlan((prev) => prev.filter((p) => p.id !== id));
  }

  function movePlan(id: string, dir: -1 | 1) {
    setPlan((prev) => {
      const idx = prev.findIndex((p) => p.id === id);
      if (idx < 0) return prev;
      const nextIdx = idx + dir;
      if (nextIdx < 0 || nextIdx >= prev.length) return prev;
      const copy = [...prev];
      const [item] = copy.splice(idx, 1);
      copy.splice(nextIdx, 0, item);
      return copy;
    });
  }

  async function runNlpParse() {
    if (!nlText.trim()) {
      setNlConfidence(null);
      setNlQuestions([]);
      setNlWarnings([]);
      return;
    }

    try {
      setStatusHint("Understanding situation…");
      const res = await postJson<NlpParseResponse>(`${API_BASE}/nlp/parse`, {
        text: nlText,
        defaults: {
          period: quarter,
          time_remaining: timeLeft,
          margin,
        },
      });

      setNlConfidence(res.confidence ?? null);
      setNlQuestions(res.clarifying_questions ?? []);
      setNlWarnings(res.warnings ?? []);

      const c = res.context ?? {};

      if (c.period !== undefined && c.period !== null) {
        setQuarter(clamp(Number(c.period), 1, 4));
      }
      if (c.time_remaining !== undefined && c.time_remaining !== null) {
        setTimeLeft(clamp(Number(c.time_remaining), 0, 720));
      }
      if (c.margin !== undefined && c.margin !== null) {
        setMargin(clamp(Number(c.margin), -30, 30));
      }

      setCtxAfterTimeout(Boolean(c.after_timeout));
      setCtxLateClock(Boolean(c.late_clock));
      setCtxNeed3(Boolean(c.need3));
      setCtxProtectLead(Boolean(c.protect_lead));
      setCtxEndQ(Boolean(c.end_of_quarter));
      setCtxVsSwitching(Boolean(c.vs_switching));

      if (c.must_stop) setFocus(70);
      if (c.need3 || c.quick2) setFocus(25);

      setStatusHint("Situation parsed ✅");
      setTimeout(() => setStatusHint(""), 900);
    } catch (e: any) {
      setStatusHint(e?.message || "Couldn’t parse the situation.");
    }
  }

  async function buildGameplan() {
    const myBuildId = Date.now();
    latestBuildRef.current = myBuildId;

    setLoading(true);
    setStatusHint("");
    setExplainMap({});

    try {
      if (nlText.trim()) {
        await runNlpParse();
      }

      // Baseline recommendations (Dataset1)
      const baseRankRaw = await baselineRank({
        season,
        our,
        opp,
        k: clamp(showPlays, 1, 12),
        wOff,
        wDef,
      });

      if (latestBuildRef.current !== myBuildId) return;

      const baseRank = Array.isArray(baseRankRaw) ? baseRankRaw : [];
      const baseline = baseRank.map((x, i) => normalizePlay(x, i, "baseline"));
      setBaselinePlays(baseline);

      // Smart recommendations (Dataset1 + context)
      let smart: NormalizedPlay[] = [];
      let smartRaw: any[] = [];

      try {
        const smartRankRaw = await contextRank({
          season,
          our,
          opp,
          margin,
          period: quarter,
          timeRemaining: timeLeft,
          k: clamp(showPlays, 1, 12),
          wOff,
        });

        smartRaw = Array.isArray(smartRankRaw) ? smartRankRaw : [];
        smart = smartRaw.map((x, i) => normalizePlay(x, i, "smart"));

        if (latestBuildRef.current !== myBuildId) return;
        setSmartPlays(smart);
      } catch {
        smart = [];
        smartRaw = [];
        if (latestBuildRef.current !== myBuildId) return;
        setSmartPlays([]);
      }

      const nextActive =
        recommendationStyle === "baseline"
          ? baseline
          : smart.length
          ? smart
          : baseline;

      setSelectedId(nextActive[0]?.id ?? null);

      // NLP explanations (optional)
      try {
        const mode: "baseline" | "context-ml" =
          recommendationStyle === "baseline" || !smartRaw.length ? "baseline" : "context-ml";

        const rankings = (mode === "context-ml" ? smartRaw : baseRank).slice(
          0,
          clamp(showPlays, 1, 12)
        );

        const explain = await postJson<NlpExplainResponse>(`${API_BASE}/nlp/explain`, {
          mode,
          context: {
            period: quarter,
            time_remaining: timeLeft,
            margin,
            after_timeout: ctxAfterTimeout,
            late_clock: ctxLateClock,
            need3: ctxNeed3,
            protect_lead: ctxProtectLead,
            end_of_quarter: ctxEndQ,
            vs_switching: ctxVsSwitching,
          },
          rankings,
          top_n: 6,
        });

        if (latestBuildRef.current !== myBuildId) return;

        const map: Record<string, ExplainItem> = {};
        for (const item of explain.explanation ?? []) {
          const key = String(item.play_type ?? "");
          if (key) map[key] = item;
        }
        setExplainMap(map);
      } catch {
        // optional
      }

      // Shot plan ranking (Dataset2) optional
      setShotLoading(true);
      setShotError(null);
      try {
        const shot = await fetchShotPlanRank({
          season,
          our,
          opp,
          k: 5,
          wOff,
        });

        if (latestBuildRef.current !== myBuildId) return;

        const types = Array.isArray(shot?.top_shot_types) ? shot.top_shot_types : [];
        const zones = Array.isArray(shot?.top_zones) ? shot.top_zones : [];

        setTopShotTypes(types);
        setTopZones(zones);

        const firstType = types[0]?.SHOT_TYPE ?? "";
        const firstZone = zones[0]?.ZONE ?? "";
        setSelectedShotType((prev) => prev || String(firstType));
        setSelectedZone((prev) => prev || String(firstZone));
      } catch (e: any) {
        if (latestBuildRef.current !== myBuildId) return;
        setTopShotTypes([]);
        setTopZones([]);
        setSelectedShotType("");
        setSelectedZone("");
        setShotError(e?.message || "Shot plan endpoint not available.");
      } finally {
        if (latestBuildRef.current === myBuildId) setShotLoading(false);
      }

      setStatusHint(smart.length ? "Smart recommendations ready ✅" : "Using baseline ✅");
      setTimeout(() => setStatusHint(""), 900);
    } catch (e: any) {
      setStatusHint(e?.message || "Couldn’t build the gameplan.");
    } finally {
      if (latestBuildRef.current === myBuildId) setLoading(false);
    }
  }

  // Fetch play zones visualization when selection changes.
  useEffect(() => {
    let alive = true;
    (async () => {
      setPlayVizError(null);
      setPlayVizBase64("");
      setPlayVizCaption("");

      if (!selectedPlay) return;

      setPlayVizLoading(true);
      try {
        const res = await fetchPlaytypeViz({
          season,
          our,
          opp,
          playType: selectedPlay.playType,
          wOff,
        });
        if (!alive) return;
        setPlayVizCaption(res?.caption ?? "");
        setPlayVizBase64(res?.image_base64 ?? "");
      } catch (e: any) {
        if (!alive) return;
        setPlayVizError(e?.message || "Couldn’t load play diagram.");
      } finally {
        if (alive) setPlayVizLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [selectedPlay?.playType, season, our, opp, wOff]);

  // Fetch shot heatmap when tab is active.
  useEffect(() => {
    let alive = true;
    (async () => {
      setShotVizError(null);
      setShotVizBase64("");
      setShotVizCaption("");

      if (vizTab !== "shotHeatmap") return;
      if (!season || !our || !opp) return;

      setShotVizLoading(true);
      try {
        const res = await fetchShotHeatmap({
          season,
          our,
          opp,
          shotType: selectedShotType || undefined,
          zone: selectedZone || undefined,
        });
        if (!alive) return;
        setShotVizCaption(res?.caption ?? "");
        setShotVizBase64(res?.image_base64 ?? "");
      } catch (e: any) {
        if (!alive) return;
        setShotVizError(e?.message || "Couldn’t load shot heatmap.");
      } finally {
        if (alive) setShotVizLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [vizTab, season, our, opp, selectedShotType, selectedZone]);

  const shotPlanBullets = useMemo(() => {
    const bullets: string[] = [];

    if (ctxNeed3) {
      bullets.push("Primary: create an open 3 (corner or slot) off a paint touch.");
    } else if (ctxProtectLead) {
      bullets.push("Primary: safe, high-quality shot (rim or catch-and-shoot). No risky passes.");
    } else {
      bullets.push("Primary: get a clean rim touch or a catch-and-shoot 3.");
    }

    if (selectedShotType) bullets.push(`Look for: ${selectedShotType}.`);
    if (selectedZone) bullets.push(`Target zone: ${selectedZone}.`);

    if (ctxVsSwitching) bullets.push("Vs switch: slip early, then attack the mismatch with pace.");
    if (ctxLateClock) bullets.push("Late clock: simplify — 1 action to the rim, 1 kick-out option.");

    if (!selectedShotType && !selectedZone) {
      bullets.push("(Shot plan will auto-fill when the Dataset2 endpoint responds.)");
    }

    return bullets;
  }, [ctxNeed3, ctxProtectLead, ctxVsSwitching, ctxLateClock, selectedShotType, selectedZone]);

  const shotTypeOptions = useMemo(() => {
    const opts = topShotTypes.map((x) => String(x?.SHOT_TYPE ?? "")).filter(Boolean);
    return Array.from(new Set(opts));
  }, [topShotTypes]);

  const zoneOptions = useMemo(() => {
    const opts = topZones.map((x) => String(x?.ZONE ?? "")).filter(Boolean);
    return Array.from(new Set(opts));
  }, [topZones]);

  const coachExplainFor = (playType: string): ExplainItem | null => {
    const item = explainMap[playType];
    return item ?? null;
  };

  return (
    <div className={styles.page}>
      {/* HERO / TOP BAR */}
      <header className={styles.hero}>
        <div className={styles.heroTop}>
          <div>
            <h1 className={styles.h1}>Gameplan</h1>
            <p className={styles.sub}>
              Set the situation → get 3 clear options → tap for diagram & counters.
            </p>
          </div>

          <div className={styles.heroActions}>
            <button
              className={styles.ghostBtn}
              type="button"
              onClick={() => window.print()}
              title="Print this page (save as PDF)"
            >
              Print / Save PDF
            </button>

            <button
              className={styles.primaryBtn}
              type="button"
              onClick={buildGameplan}
              disabled={loading}
            >
              {loading ? "Building…" : "Build Gameplan"}
            </button>
          </div>
        </div>

        {/* Natural language input */}
        <div className={styles.panel} style={{ marginTop: 14 }}>
          <div className={styles.panelHeader}>
            <h2 className={styles.h2}>Describe the situation</h2>
            <span className={styles.smallMuted}>
              Type it like a coach: score, time, any constraints.
            </span>
          </div>

          <textarea
            className={styles.textarea}
            value={nlText}
            onChange={(e) => setNlText(e.target.value)}
            placeholder="Example: 'Tied, 0:38 left in Q4. Need a good look. They are switching.'"
          />

          <div className={styles.cardBtns}>
            <button className={styles.btnSoft} type="button" onClick={runNlpParse}>
              Understand
            </button>
            <button
              className={styles.btnSoft}
              type="button"
              onClick={() => {
                setNlText("");
                setNlConfidence(null);
                setNlQuestions([]);
                setNlWarnings([]);
                setStatusHint("Cleared.");
                setTimeout(() => setStatusHint(""), 800);
              }}
            >
              Clear
            </button>

            {nlConfidence !== null ? (
              <span className={styles.smallMuted}>
                Confidence: {(nlConfidence * 100).toFixed(0)}%
              </span>
            ) : null}

            {statusHint ? <span className={styles.statusHint}>{statusHint}</span> : null}
          </div>

          {nlWarnings.length ? (
            <div className={styles.warn}>
              {nlWarnings.slice(0, 2).map((w, i) => (
                <div key={i} className={styles.warnSmall}>
                  {w}
                </div>
              ))}
            </div>
          ) : null}

          {nlQuestions.length ? (
            <div className={styles.warn}>
              <div className={styles.warnSmall}>
                Quick questions: {nlQuestions.slice(0, 2).join(" • ")}
              </div>
            </div>
          ) : null}
        </div>

        {/* Situation controls */}
        <div className={styles.situationBar}>
          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Season</div>
            <select
              className={styles.select}
              value={season}
              onChange={(e) => setSeason(e.target.value)}
            >
              {seasonOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Our team</div>
            <select className={styles.select} value={our} onChange={(e) => setOur(e.target.value)}>
              {(teamOptions.length ? teamOptions : [{ code: "TOR" }]).map((t) => (
                <option key={t.code} value={t.code}>
                  {t.code}
                  {t.name ? ` — ${t.name}` : ""}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Opponent</div>
            <select className={styles.select} value={opp} onChange={(e) => setOpp(e.target.value)}>
              {(teamOptions.length ? teamOptions : [{ code: "BOS" }]).map((t) => (
                <option key={t.code} value={t.code}>
                  {t.code}
                  {t.name ? ` — ${t.name}` : ""}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlockWide}>
            <div className={styles.sitLabel}>
              Focus <span className={styles.sitHint}>Need a bucket ↔ Must get a stop</span>
            </div>
            <input
              className={styles.slider}
              type="range"
              min={0}
              max={100}
              step={5}
              value={focus}
              onChange={(e) => setFocus(Number(e.target.value))}
            />
            <div className={styles.sliderEnds}>
              <span>Bucket</span>
              <span>Stop</span>
            </div>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Plays shown</div>
            <select
              className={styles.select}
              value={showPlays}
              onChange={(e) => setShowPlays(Number(e.target.value))}
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Recommendations</div>
            <select
              className={styles.select}
              value={recommendationStyle}
              onChange={(e) => setRecommendationStyle(e.target.value as "smart" | "baseline")}
            >
              <option value="smart">Smart (ML + context)</option>
              <option value="baseline">Baseline</option>
            </select>
          </div>
        </div>

        {/* Context row */}
        <div className={styles.contextRow}>
          <div className={styles.contextLeft}>
            <div className={styles.contextTitle}>Context</div>
            <div className={styles.chips}>
              <Chip active={ctxAfterTimeout} label="After timeout" onClick={() => setCtxAfterTimeout((v) => !v)} />
              <Chip active={ctxLateClock} label="Late clock" onClick={() => setCtxLateClock((v) => !v)} />
              <Chip active={ctxNeed3} label="Need a 3" onClick={() => setCtxNeed3((v) => !v)} />
              <Chip active={ctxProtectLead} label="Protect lead" onClick={() => setCtxProtectLead((v) => !v)} />
              <Chip active={ctxEndQ} label="End of quarter" onClick={() => setCtxEndQ((v) => !v)} />
              <Chip active={ctxVsSwitching} label="Vs switching" onClick={() => setCtxVsSwitching((v) => !v)} />
            </div>
          </div>

          <div className={styles.contextRight}>
            <div className={styles.miniInputs}>
              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Quarter</div>
                <input
                  className={styles.input}
                  type="number"
                  min={1}
                  max={4}
                  value={quarter}
                  onChange={(e) => setQuarter(clamp(Number(e.target.value || 4), 1, 4))}
                />
              </div>

              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Time left (sec)</div>
                <input
                  className={styles.input}
                  type="number"
                  min={0}
                  max={720}
                  value={timeLeft}
                  onChange={(e) => setTimeLeft(clamp(Number(e.target.value || 180), 0, 720))}
                />
              </div>

              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Score (our − opp)</div>
                <input
                  className={styles.input}
                  type="number"
                  min={-30}
                  max={30}
                  value={margin}
                  onChange={(e) => setMargin(clamp(Number(e.target.value || 0), -30, 30))}
                />
              </div>
            </div>

            <div className={styles.situationSummary}>
              <span className={styles.summaryBadge}>{situationSummary}</span>
            </div>
          </div>
        </div>

        {optError ? (
          <div className={styles.warn}>
            Couldn’t load team/season options. (Backend might be off.) You can still try building.
            <div className={styles.warnSmall}>{optError}</div>
          </div>
        ) : null}
      </header>

      {/* MAIN GRID */}
      <div className={styles.grid}>
        {/* LEFT: Quick Call + Plan */}
        <aside className={styles.left}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Quick Call</h2>
              <span className={styles.smallMuted}>The “what to run next” board.</span>
            </div>

            {!top3.length ? (
              <div className={styles.empty}>
                Tap <b>Build Gameplan</b> to get your options.
              </div>
            ) : (
              <div className={styles.quickCards}>
                {top3.map((p, idx) => {
                  const grade = gradeFromRank(idx);
                  const isActive = selectedPlay?.id === p.id;

                  const headline = idx === 0 ? "Primary" : idx === 1 ? "Secondary" : "Safety";

                  const coachExplain = coachExplainFor(p.playType);
                  const whyBullets =
                    coachExplain?.evidence?.length
                      ? coachExplain.evidence
                      : p.why.length
                      ? p.why
                      : ["Strong fit for your matchup + focus settings."];

                  return (
                    <div
                      key={p.id}
                      className={`${styles.quickCard} ${isActive ? styles.quickCardActive : ""}`}
                    >
                      <div className={styles.quickTop}>
                        <div className={styles.quickLeft}>
                          <div className={styles.quickLabel}>{headline}</div>
                          <div className={styles.quickName}>{p.playType}</div>
                        </div>

                        <div className={`${styles.grade} ${styles["grade" + grade]}`}>{grade}</div>
                      </div>

                      <div className={styles.quickStats}>
                        <div className={styles.stat}>
                          <div className={styles.statLabel}>Points / play</div>
                          <div className={styles.statValue}>
                            {p.ppp !== null ? p.ppp.toFixed(3) : "—"}
                          </div>
                        </div>
                        <div className={styles.stat}>
                          <div className={styles.statLabel}>Edge</div>
                          <div className={styles.statValue}>
                            {p.edge !== null ? `${p.edge >= 0 ? "+" : ""}${p.edge.toFixed(3)}` : "—"}
                          </div>
                        </div>
                      </div>

                      <div className={styles.quickWhy}>
                        <div className={styles.smallMuted}>Why it works</div>
                        {coachExplain?.summary ? (
                          <div className={styles.warnSmall} style={{ marginBottom: 8 }}>
                            {coachExplain.summary}
                          </div>
                        ) : null}
                        <ul className={styles.bullets}>
                          {whyBullets.slice(0, 3).map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                        {coachExplain?.caution ? (
                          <div className={styles.warnSmall} style={{ marginTop: 10 }}>
                            <b>Watch:</b> {coachExplain.caution}
                          </div>
                        ) : null}
                      </div>

                      <div className={styles.quickHow}>
                        <div className={styles.smallMuted}>How to run</div>
                        <ol className={styles.steps}>
                          {coachingCuesFor(p.playType).slice(0, 3).map((c, i) => (
                            <li key={i}>{c}</li>
                          ))}
                        </ol>
                      </div>

                      <div className={styles.cardBtns}>
                        <button
                          className={styles.btnSoft}
                          type="button"
                          onClick={() => {
                            setSelectedId(p.id);
                            setVizTab("playZones");
                          }}
                        >
                          Diagram
                        </button>
                        <button
                          className={styles.btnSoft}
                          type="button"
                          onClick={() => {
                            setSelectedId(p.id);
                            const el = document.getElementById("counters");
                            el?.scrollIntoView({ behavior: "smooth", block: "start" });
                          }}
                        >
                          Counters
                        </button>
                        <button
                          className={styles.btnPrimarySmall}
                          type="button"
                          onClick={() => addToPlan(p)}
                        >
                          Add to plan
                        </button>
                      </div>

                      <button
                        type="button"
                        className={styles.selectCardOverlay}
                        onClick={() => setSelectedId(p.id)}
                        aria-label={`Select ${p.playType}`}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Your Plan</h2>
              <span className={styles.smallMuted}>For the huddle / timeout.</span>
            </div>

            {!plan.length ? (
              <div className={styles.empty}>
                Add plays from <b>Quick Call</b>.
              </div>
            ) : (
              <div className={styles.planList}>
                {plan.map((p) => (
                  <div key={p.id} className={styles.planItem}>
                    <div className={styles.planLabel}>{p.label}</div>
                    <div className={styles.planBtns}>
                      <button className={styles.iconBtn} type="button" onClick={() => movePlan(p.id, -1)} title="Move up">
                        ↑
                      </button>
                      <button className={styles.iconBtn} type="button" onClick={() => movePlan(p.id, 1)} title="Move down">
                        ↓
                      </button>
                      <button className={styles.iconBtnDanger} type="button" onClick={() => removeFromPlan(p.id)} title="Remove">
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className={styles.planActions}>
              <button
                className={styles.ghostBtn}
                type="button"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(window.location.href);
                    setStatusHint("Link copied ✅");
                    setTimeout(() => setStatusHint(""), 900);
                  } catch {
                    setStatusHint("Copy failed — copy from the address bar.");
                    setTimeout(() => setStatusHint(""), 1200);
                  }
                }}
              >
                Copy link
              </button>

              <button className={styles.primaryBtn} type="button" onClick={() => window.print()}>
                Print / Save PDF
              </button>
            </div>
          </section>
        </aside>

        {/* RIGHT: Details */}
        <main className={styles.right}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Shot Plan</h2>
              <span className={styles.smallMuted}>Simple “what shot we want” guidance.</span>
            </div>

            <div className={styles.shotPlan}>
              <div className={styles.shotHeadline}>
                {ctxNeed3 ? "We want an open 3…" : "We want a high-quality shot…"}
              </div>

              <ul className={styles.bullets}>
                {shotPlanBullets.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>

              <div className={styles.shotBtns}>
                <button
                  className={styles.btnSoft}
                  type="button"
                  onClick={() => setVizTab("shotHeatmap")}
                  disabled={shotLoading}
                >
                  {shotLoading ? "Loading…" : "Show heatmap"}
                </button>

                <button
                  className={styles.btnSoft}
                  type="button"
                  onClick={() => authenticatedDownload(getShotPlanPdfUrl({
                    season,
                    our,
                    opp,
                    k: 5,
                    wOff,
                    shotType: selectedShotType || undefined,
                    zone: selectedZone || undefined,
                  })).catch(e => alert(e))}
                >
                  Export PDF
                </button>
              </div>

              <div className={styles.miniInputs} style={{ marginTop: 10 }}>
                <div className={styles.miniBlock}>
                  <div className={styles.sitLabel}>Shot type</div>
                  <select
                    className={styles.select}
                    value={selectedShotType}
                    onChange={(e) => setSelectedShotType(e.target.value)}
                    disabled={!shotTypeOptions.length}
                  >
                    <option value="">(Any)</option>
                    {shotTypeOptions.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.miniBlock}>
                  <div className={styles.sitLabel}>Zone</div>
                  <select
                    className={styles.select}
                    value={selectedZone}
                    onChange={(e) => setSelectedZone(e.target.value)}
                    disabled={!zoneOptions.length}
                  >
                    <option value="">(Any)</option>
                    {zoneOptions.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {shotError ? <div className={styles.warnSmall}>{shotError}</div> : null}
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Court View</h2>
              <span className={styles.smallMuted}>Visual only — no clutter.</span>
            </div>

            <div className={styles.tabs}>
              <button
                type="button"
                className={`${styles.tab} ${vizTab === "playZones" ? styles.tabActive : ""}`}
                onClick={() => setVizTab("playZones")}
              >
                Play zones
              </button>

              <button
                type="button"
                className={`${styles.tab} ${vizTab === "shotHeatmap" ? styles.tabActive : ""}`}
                onClick={() => setVizTab("shotHeatmap")}
              >
                Shot heatmap
              </button>
            </div>

            {vizTab === "playZones" ? (
              selectedPlay ? (
                <div className={styles.vizWrap}>
                  <div className={styles.vizTitle}>
                    Diagram for: <b>{selectedPlay.playType}</b>
                  </div>

                  {playVizLoading ? (
                    <div className={styles.empty}>Loading diagram…</div>
                  ) : playVizError ? (
                    <div className={styles.warnSmall}>{playVizError}</div>
                  ) : playVizBase64 ? (
                    <>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        className={styles.vizImg}
                        src={`data:image/png;base64,${playVizBase64}`}
                        alt={`Play zones for ${selectedPlay.playType}`}
                      />
                      {playVizCaption ? (
                        <div className={styles.vizCaption}>{playVizCaption}</div>
                      ) : null}
                    </>
                  ) : (
                    <div className={styles.empty}>Build a gameplan to see visuals.</div>
                  )}
                </div>
              ) : (
                <div className={styles.empty}>Build a gameplan to see visuals.</div>
              )
            ) : (
              <div className={styles.vizWrap}>
                <div className={styles.vizTitle}>Shot Heatmap</div>

                {shotVizLoading ? (
                  <div className={styles.empty}>Loading heatmap…</div>
                ) : shotVizError ? (
                  <div className={styles.warnSmall}>{shotVizError}</div>
                ) : shotVizBase64 ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img className={styles.vizImg} src={`data:image/png;base64,${shotVizBase64}`} alt="Shot heatmap" />
                    {shotVizCaption ? (
                      <div className={styles.vizCaption}>{shotVizCaption}</div>
                    ) : null}
                  </>
                ) : (
                  <div className={styles.empty}>Build a gameplan to fetch shot data, then open this tab.</div>
                )}
              </div>
            )}
          </section>

          <section className={styles.panel} id="counters">
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Counters</h2>
              <span className={styles.smallMuted}>“If they adjust… here’s what we do.”</span>
            </div>

            {!selectedPlay ? (
              <div className={styles.empty}>Select a play to see counters.</div>
            ) : (
              <div className={styles.counterGrid}>
                {counterPlanFor(selectedPlay.playType).map((c) => (
                  <div key={c.title} className={styles.counterCard}>
                    <div className={styles.counterTitle}>{c.title}</div>
                    <div className={styles.counterTrigger}>{c.trigger}</div>
                    <ul className={styles.bullets}>
                      {c.cues.map((x, i) => (
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                    <div className={styles.counterOutcome}>{c.outcome}</div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Assignments</h2>
              <span className={styles.smallMuted}>Player roles (fill in names for your lineup).</span>
            </div>

            <div className={styles.roles}>
              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Ball Handler</div>
                <input className={styles.input} placeholder="Name" value={roles.ballHandler} onChange={(e) => setRoles({ ...roles, ballHandler: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Screener / Roller</div>
                <input className={styles.input} placeholder="Name" value={roles.screener} onChange={(e) => setRoles({ ...roles, screener: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Corner Spacer</div>
                <input className={styles.input} placeholder="Name" value={roles.cornerSpacer} onChange={(e) => setRoles({ ...roles, cornerSpacer: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Weak-side Cutter</div>
                <input className={styles.input} placeholder="Name" value={roles.cutter} onChange={(e) => setRoles({ ...roles, cutter: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Safety Outlet</div>
                <input className={styles.input} placeholder="Name" value={roles.safety} onChange={(e) => setRoles({ ...roles, safety: e.target.value })} />
              </div>

              <div className={styles.smallMuted}>
                Tip: Keep the “safety” player high for spacing + transition defense.
              </div>
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Notes</h2>
              <span className={styles.smallMuted}>Keep it simple: reminders for the huddle.</span>
            </div>

            {!selectedPlay ? (
              <div className={styles.empty}>Select a play to add notes.</div>
            ) : (
              <div className={styles.notesBox}>
                <div className={styles.noteTitle}>
                  Notes for: <b>{selectedPlay.playType}</b>
                </div>

                <textarea
                  className={styles.textarea}
                  placeholder="Example: ‘If they switch, slip it. Corners stay lifted. First look: roll.’"
                  value={notesByPlay[selectedPlay.id] ?? ""}
                  onChange={(e) =>
                    setNotesByPlay((prev) => ({
                      ...prev,
                      [selectedPlay.id]: e.target.value,
                    }))
                  }
                />

                <details className={styles.details}>
                  <summary className={styles.summary}>More details (optional)</summary>
                  <div className={styles.detailsBody}>
                    <div className={styles.detailRow}>
                      <span className={styles.detailKey}>Internal focus</span>
                      <span className={styles.detailVal}>
                        Offense {wOff.toFixed(2)} / Defense {wDef.toFixed(2)}
                      </span>
                    </div>
                    <div className={styles.detailRow}>
                      <span className={styles.detailKey}>Data note</span>
                      <span className={styles.detailVal}>
                        We keep the UI coach-friendly — deep stats stay in the Data Explorer.
                      </span>
                    </div>
                  </div>
                </details>
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
