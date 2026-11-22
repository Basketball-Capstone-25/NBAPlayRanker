// Gameplan page UI.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Gameplan.module.css";
import { gameplanSelectionKey, selectGameplanResult, type GameplanSelection } from "./gameplan-result";
import TopPickCall from "../../_shared/components/TopPickCall";
import { periodLabel, periodSeconds, resolveGameSituation, type ParsedSituation, type ResolvedSituation } from "../../../domain/gameplan-context";
import { FIELD_LABEL, situationFromParse, type ParseReply, type SituationField } from "../../../domain/voice-situation";

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
import {
  EMPTY_ROLES,
  SYNC_LABELS,
  loadGameplan,
  readLegacyLocal,
  saveGameplan,
  supabaseGameplanBackend,
  type GameplanState,
  type SyncStatus,
} from "../../../infrastructure/gameplan-sync";

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

export function GameplanShotAdvice({ shotType, zone }: { shotType: string; zone: string }) {
  return <>
    <div className={styles.shotHeadline}>Shot priorities from season data</div>
    <p className={styles.smallMuted}>
      These rates compare season-long offense and opponent defense. They do not use the score, clock or coaching conditions.
      Estimates are points per shot attempt, not points per possession.
    </p>
    <ul className={styles.bullets}>
      {shotType ? <li>Selected shot type: {shotType}.</li> : null}
      {zone ? <li>Selected court zone: {zone}.</li> : null}
    </ul>
  </>;
}

export function GameplanDiagramCaption({ caption }: { caption: string }) {
  return <div className={styles.vizCaption}>
    <div>{caption}</div>
    <div>Season-baseline diagram. Its PPP does not include the game situation or coaching adjustments.</div>
  </div>;
}

export default function GameplanClient() {
  const [options, setOptions] = useState<MetaOptions | null>(null);
  const [optError, setOptError] = useState<string | null>(null);

  // Situation
  // SCRUM-841: the coach chooses the matchup; nothing is pre-selected.
  const [season, setSeason] = useState("");
  const [our, setOur] = useState("");
  const [opp, setOpp] = useState("");

  // Friendly focus slider (0..100)
  const [focus, setFocus] = useState(35);

  // Context
  // SCRUM-841: the game situation starts blank; the coach sets it or describes it.
  const [quarter, setQuarter] = useState<number | null>(null);
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const [margin, setMargin] = useState<number | null>(null);
  const situationSet = quarter !== null && timeLeft !== null && margin !== null;

  const [ctxAfterTimeout, setCtxAfterTimeout] = useState(false);
  const [ctxLateClock, setCtxLateClock] = useState(false);
  const [ctxNeed3, setCtxNeed3] = useState(false);
  const [ctxProtectLead, setCtxProtectLead] = useState(false);
  const [ctxEndQ, setCtxEndQ] = useState(false);
  const [ctxVsSwitching, setCtxVsSwitching] = useState(false);

  // Natural language (NLP)
  const [nlText, setNlText] = useState<string>("");
  const [nlConfidence, setNlConfidence] = useState<number | null>(null);
  const [nlQuestions, setNlQuestions] = useState<string[]>([]);
  const [nlWarnings, setNlWarnings] = useState<string[]>([]);
  const [parsedConditions, setParsedConditions] = useState<ParsedSituation>({});
  const lastParsedTextRef = useRef("");
  const [rankingNote, setRankingNote] = useState("");
  const [parsing, setParsing] = useState(false);

  // Controls
  const [showPlays, setShowPlays] = useState(5);
  const [recommendationStyle, setRecommendationStyle] = useState<"smart" | "baseline">("smart");

  // Results
  const [loading, setLoading] = useState(false);
  const [baselinePlays, setBaselinePlays] = useState<NormalizedPlay[]>([]);
  const [smartPlays, setSmartPlays] = useState<NormalizedPlay[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [completedSelection, setCompletedSelection] = useState<GameplanSelection | null>(null);

  // NLP explanations for top plays
  const [explainMap, setExplainMap] = useState<Record<string, ExplainItem>>({});

  // Visuals
  const [vizTab, setVizTab] = useState<"playZones" | "shotHeatmap">("playZones");
  const [playVizLoading, setPlayVizLoading] = useState(false);
  const [playVizError, setPlayVizError] = useState<string | null>(null);
  const [playVizCaption, setPlayVizCaption] = useState<string>("");
  const [playVizBase64, setPlayVizBase64] = useState<string>("");
  const [loadedPlayVizKey, setLoadedPlayVizKey] = useState("");

  const [shotVizLoading, setShotVizLoading] = useState(false);
  const [shotVizError, setShotVizError] = useState<string | null>(null);
  const [shotVizCaption, setShotVizCaption] = useState<string>("");
  const [shotVizBase64, setShotVizBase64] = useState<string>("");
  const [loadedShotVizKey, setLoadedShotVizKey] = useState("");

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

  // SCRUM-483: Supabase sync with loading/saving states and a local fallback.
  const syncBackend = useMemo(() => supabaseGameplanBackend(), []);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("loading");
  const [syncNote, setSyncNote] = useState<string>("");
  const [offerDraft, setOfferDraft] = useState<{ state: GameplanState; label: string } | null>(null);
  const revisionRef = useRef<number | null>(null);
  const loadedKeyRef = useRef<string>("");
  const lastSyncedRef = useRef<string>("");

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

        // Keep a choice only if it exists in the loaded data.
        const seasons = data?.seasons ?? [];
        const codes = new Set(normalizeTeams(data).map((t) => t.code));
        setSeason((prev) => (seasons.includes(prev) ? prev : ""));
        setOur((prev) => (codes.has(prev) ? prev : ""));
        setOpp((prev) => (codes.has(prev) ? prev : ""));
      } catch (e: any) {
        if (!alive) return;
        setOptError(e?.message || "Couldn’t load team/season options.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const matchup = useMemo(() => ({ season, ourTeam: our, oppTeam: opp }), [season, our, opp]);
  const matchupKey = `${season}|${our}|${opp}`;

  function applyState(next: GameplanState) {
    setNotesByPlay(next.notes);
    setPlan(next.plan);
    setRoles(next.roles);
  }

  // Load the coach's saved plan for this matchup; fall back to this device.
  useEffect(() => {
    let alive = true;
    loadedKeyRef.current = "";
    if (!season || !our || !opp) return () => { alive = false; };
    setSyncStatus("loading");
    setSyncNote("");
    setOfferDraft(null);
    (async () => {
      const result = await loadGameplan(syncBackend, matchup);
      if (!alive) return;
      const legacy = readLegacyLocal(typeof window === "undefined" ? null : window.localStorage);
      if (result.status === "saved") {
        const loaded = result.state ?? { notes: {}, plan: [], roles: EMPTY_ROLES };
        revisionRef.current = result.revision;
        applyState(loaded);
        lastSyncedRef.current = JSON.stringify(loaded);
        // Never upload old device notes silently: offer an explicit import.
        if (!result.state && legacy) setOfferDraft({ state: legacy, label: "Import notes saved on this device" });
        setSyncStatus("saved");
      } else {
        revisionRef.current = null;
        if (legacy) applyState(legacy);
        setSyncNote(result.reason);
        setSyncStatus("local");
      }
      loadedKeyRef.current = matchupKey;
    })();
    return () => {
      alive = false;
    };
  }, [syncBackend, matchup, matchupKey]);

  async function syncNow() {
    const key = matchupKey;
    if (loadedKeyRef.current !== key) return; // never write one matchup's draft into another
    const draft: GameplanState = { notes: notesByPlay, plan, roles };
    const json = JSON.stringify(draft);
    setSyncStatus("saving");
    const result = await saveGameplan(syncBackend, matchup, draft, revisionRef.current);
    if (loadedKeyRef.current !== key) return;
    if (result.status === "saved") {
      revisionRef.current = result.revision;
      lastSyncedRef.current = json;
      setSyncNote("");
      setSyncStatus("saved");
    } else if (result.status === "conflict") {
      revisionRef.current = result.revision;
      if (result.state) {
        applyState(result.state);
        lastSyncedRef.current = JSON.stringify(result.state);
      }
      setOfferDraft({ state: draft, label: "Restore my unsaved edits" });
      setSyncStatus("conflict");
    } else if (result.status === "local") {
      setSyncNote(result.reason);
      setSyncStatus("local");
    } else {
      setSyncNote(result.message);
      setSyncStatus("error");
    }
  }

  // Debounced save after edits once the matchup has loaded.
  useEffect(() => {
    if (loadedKeyRef.current !== matchupKey) return;
    if (syncStatus === "loading" || syncStatus === "local") return;
    if (JSON.stringify({ notes: notesByPlay, plan, roles }) === lastSyncedRef.current) return;
    const timer = setTimeout(() => void syncNow(), 800);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notesByPlay, plan, roles]);

  // The localStorage copy below stays as this device's fallback.
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
  const seasonOptions = useMemo(() => options?.seasons ?? [], [options]);
  const matchupChosen = Boolean(season && our && opp && our !== opp);

  const draftSelection: GameplanSelection = {
    season, our, opp, k: clamp(showPlays, 1, 10), wOff, wDef,
    situation: situationSet ? currentSituation() : null, prompt: nlText,
  };
  const currentInputKey = gameplanSelectionKey(draftSelection);
  const currentInputKeyRef = useRef(currentInputKey);
  currentInputKeyRef.current = currentInputKey;
  const resultState = selectGameplanResult(completedSelection, draftSelection, {
    loading, parsing, shotLoading, shotError, hasShots: topShotTypes.length > 0 || topZones.length > 0,
  });
  const resultInputs = resultState.inputs;

  const activePlays = useMemo(() => {
    if (!resultState.current) return [];
    if (recommendationStyle === "baseline") return baselinePlays;
    return smartPlays.length ? smartPlays : baselinePlays;
  }, [resultState.current, recommendationStyle, baselinePlays, smartPlays]);

  const selectedPlay = useMemo(() => {
    if (!activePlays.length) return null;
    if (!selectedId) return activePlays[0];
    return activePlays.find((p) => p.id === selectedId) ?? activePlays[0];
  }, [activePlays, selectedId]);

  const top3 = useMemo(() => activePlays.slice(0, 3), [activePlays]);
  const playVizKey = resultInputs && selectedPlay ? `${gameplanSelectionKey(resultInputs)}|${selectedPlay.playType}` : "";
  const shotVizKey = resultState.shotsCurrent && resultInputs ? `${gameplanSelectionKey(resultInputs)}|${selectedShotType}|${selectedZone}` : "";

  const situationSummary = useMemo(() => {
    if (quarter === null || timeLeft === null || margin === null) return "Situation not set";
    const upDown =
      margin === 0 ? "Tied" : margin > 0 ? `Up ${margin}` : `Down ${Math.abs(margin)}`;
    const t = prettyTime(timeLeft);
    return `${periodLabel(quarter)} • ${t} left • ${upDown}`;
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

  /** Only called once the period, time left and score are all set. */
  function currentSituation(): ResolvedSituation {
    const period = quarter ?? 0, time_remaining = timeLeft ?? 0, score = margin ?? 0;
    return resolveGameSituation({
      ...parsedConditions,
      period, time_remaining, margin: score,
      after_timeout: ctxAfterTimeout, late_clock: ctxLateClock, need3: ctxNeed3,
      protect_lead: ctxProtectLead, end_of_quarter: ctxEndQ, vs_switching: ctxVsSwitching,
      need: ctxNeed3 ? "need3" : parsedConditions.need === "need3" ? undefined : parsedConditions.need,
      defense_style: ctxVsSwitching ? "switch" : parsedConditions.defense_style === "switch" ? undefined : parsedConditions.defense_style,
    }, { period, time_remaining, margin: score });
  }

  // SCRUM-841: only values the description actually gives are filled; the rest
  // keep what the coach set (or stay blank). Nothing is guessed.
  async function runNlpParse(): Promise<ResolvedSituation | null> {
    if (!nlText.trim()) {
      setNlConfidence(null);
      setNlQuestions([]);
      setNlWarnings([]);
      return situationSet ? currentSituation() : null;
    }

    const parseKey = currentInputKeyRef.current;
    try {
      setParsing(true);
      setStatusHint("Understanding situation…");
      const res = await postJson<NlpParseResponse & ParseReply>(`${API_BASE}/nlp/parse`, { text: nlText });
      if (currentInputKeyRef.current !== parseKey) {
        setStatusHint("Inputs changed. Build the gameplan again.");
        return null;
      }
      setNlConfidence(res.confidence ?? null);
      setNlQuestions(res.clarifying_questions ?? []);
      const c = res.context ?? {};
      setParsedConditions(c);
      lastParsedTextRef.current = nlText;

      const heard = situationFromParse(res);
      const period = heard.period ?? quarter;
      const seconds = heard.timeRemaining ?? timeLeft;
      const score = heard.margin ?? margin;
      setQuarter(period);
      setTimeLeft(seconds);
      setMargin(score);
      if (period === null || seconds === null || score === null) {
        const missing: SituationField[] = [];
        if (score === null) missing.push("margin");
        if (period === null) missing.push("period");
        if (seconds === null) missing.push("timeRemaining");
        const names = missing.map((f) => FIELD_LABEL[f]).join(" and ");
        setNlWarnings([...(res.warnings ?? []), `The description did not give the ${names}. Set it below.`]);
        setStatusHint(`Set the ${names} to continue.`);
        return null;
      }

      const resolved = resolveGameSituation(c, { period, time_remaining: seconds, margin: score });
      setNlWarnings([...resolved.warnings, ...(res.warnings ?? [])]);
      setQuarter(resolved.period);
      setTimeLeft(resolved.time_remaining);
      setMargin(resolved.margin);
      const coaching = resolved.coaching;
      setCtxAfterTimeout(Boolean(coaching.after_timeout));
      setCtxLateClock(Boolean(coaching.late_clock));
      setCtxNeed3(Boolean(coaching.need3 || coaching.need === "need3"));
      setCtxProtectLead(Boolean(coaching.protect_lead));
      setCtxEndQ(resolved.end_of_quarter);
      setCtxVsSwitching(Boolean(coaching.vs_switching || coaching.defense_style === "switch"));

      setStatusHint("Situation parsed ✅");
      setTimeout(() => setStatusHint(""), 900);
      return resolved;
    } catch (e: any) {
      setStatusHint(e?.message || "Couldn’t parse the situation.");
      throw e;
    } finally {
      setParsing(false);
    }
  }

  async function buildGameplan() {
    const myBuildId = ++latestBuildRef.current;

    setLoading(true);
    setCompletedSelection(null);
    setBaselinePlays([]);
    setSmartPlays([]);
    setTopShotTypes([]);
    setTopZones([]);
    setShotError(null);
    setStatusHint("");
    setExplainMap({});
    setRankingNote("");

    try {
      const situation = nlText.trim() && nlText !== lastParsedTextRef.current
        ? await runNlpParse()
        : situationSet ? currentSituation() : null;
      if (!situation) {
        setStatusHint("Set the period, time left and score (or describe them) before building.");
        return;
      }

      const submitted: GameplanSelection = { ...draftSelection, situation };

      // Baseline recommendations (Dataset1)
      const baseRankRaw = await baselineRank({
        season,
        our,
        opp,
        k: clamp(showPlays, 1, 10),
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
          margin: situation.margin,
          period: situation.period,
          timeRemaining: situation.time_remaining,
          coaching: situation.coaching,
          k: clamp(showPlays, 1, 10),
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
        setRankingNote("Context ranking could not load. These baseline results do not use score, period, clock or coaching conditions. Try Build again.");
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
        const mode: "baseline" | "context-ml" = smartRaw.length ? "context-ml" : "baseline";

        const rankings = (mode === "context-ml" ? smartRaw : baseRank).slice(
          0,
          clamp(showPlays, 1, 10)
        );

        if (!smartRaw.length) throw new Error("Baseline uses its own context-free explanation.");
        const explain = await postJson<NlpExplainResponse>(`${API_BASE}/nlp/explain`, {
          mode,
          context: {
            period: situation.period,
            time_remaining: situation.time_remaining,
            margin: situation.margin,
            ...situation.coaching,
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
        setSelectedShotType(String(firstType));
        setSelectedZone(String(firstZone));
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

      if (latestBuildRef.current !== myBuildId) return;
      setCompletedSelection(submitted);
      setStatusHint(smart.length ? "Smart recommendations ready ✅" : "Using baseline ✅");
      setTimeout(() => setStatusHint(""), 900);
    } catch (e: any) {
      if (latestBuildRef.current === myBuildId) setStatusHint(e?.message || "Couldn’t build the gameplan.");
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
      setPlayVizLoading(false);
      setLoadedPlayVizKey("");

      if (!selectedPlay || !resultInputs) return;

      setPlayVizLoading(true);
      try {
        const res = await fetchPlaytypeViz({
          season: resultInputs.season,
          our: resultInputs.our,
          opp: resultInputs.opp,
          playType: selectedPlay.playType,
          wOff: resultInputs.wOff,
        });
        if (!alive) return;
        setLoadedPlayVizKey(playVizKey);
        setPlayVizCaption(res?.caption ?? "");
        setPlayVizBase64(res?.image_base64 ?? "");
      } catch (e: any) {
        if (!alive) return;
        setLoadedPlayVizKey(playVizKey);
        setPlayVizError(e?.message || "Couldn’t load play diagram.");
      } finally {
        if (alive) setPlayVizLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [playVizKey, resultInputs, selectedPlay?.playType]);

  // Fetch shot heatmap when tab is active.
  useEffect(() => {
    let alive = true;
    (async () => {
      setShotVizError(null);
      setShotVizBase64("");
      setShotVizCaption("");
      setShotVizLoading(false);
      setLoadedShotVizKey("");

      if (vizTab !== "shotHeatmap" || !shotVizKey || !resultInputs) return;

      setShotVizLoading(true);
      try {
        const res = await fetchShotHeatmap({
          season: resultInputs.season,
          our: resultInputs.our,
          opp: resultInputs.opp,
          shotType: selectedShotType || undefined,
          zone: selectedZone || undefined,
        });
        if (!alive) return;
        setLoadedShotVizKey(shotVizKey);
        setShotVizCaption(res?.caption ?? "");
        setShotVizBase64(res?.image_base64 ?? "");
      } catch (e: any) {
        if (!alive) return;
        setLoadedShotVizKey(shotVizKey);
        setShotVizError(e?.message || "Couldn’t load shot heatmap.");
      } finally {
        if (alive) setShotVizLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [vizTab, shotVizKey, resultInputs, selectedShotType, selectedZone]);

  const shotTypeOptions = useMemo(() => {
    const opts = topShotTypes.map((x) => String(x?.SHOT_TYPE ?? "")).filter(Boolean);
    return Array.from(new Set(opts));
  }, [topShotTypes]);

  const zoneOptions = useMemo(() => {
    const opts = topZones.map((x) => String(x?.ZONE ?? "")).filter(Boolean);
    return Array.from(new Set(opts));
  }, [topZones]);

  const coachExplainFor = (playType: string): ExplainItem | null => {
    if (recommendationStyle === "baseline" || !smartPlays.length) return null;
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
              disabled={loading || parsing || !matchupChosen}
              title={matchupChosen ? undefined : "Choose a season and two different teams first."}
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
            aria-label="Describe the game situation"
            value={nlText}
            onChange={(e) => setNlText(e.target.value)}
            placeholder="Example: 'Tied, 0:38 left in Q4. Need a good look. They are switching.'"
          />

          <div className={styles.cardBtns}>
            <button className={styles.btnSoft} type="button" disabled={loading || parsing} onClick={() => { void runNlpParse().catch(() => {}); }}>
              {parsing ? "Understanding…" : "Understand"}
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
              {nlWarnings.map((w, i) => (
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
              aria-label="Season"
              value={season}
              onChange={(e) => setSeason(e.target.value)}
            >
              <option value="" disabled>Choose a season</option>
              {seasonOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Our team</div>
            <select className={styles.select} aria-label="Our team" value={our} onChange={(e) => setOur(e.target.value)}>
              <option value="" disabled>Choose our team</option>
              {teamOptions.map((t) => (
                <option key={t.code} value={t.code}>
                  {t.code}
                  {t.name ? ` — ${t.name}` : ""}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Opponent</div>
            <select className={styles.select} aria-label="Opponent" value={opp} onChange={(e) => setOpp(e.target.value)}>
              <option value="" disabled>Choose an opponent</option>
              {teamOptions.map((t) => (
                <option key={t.code} value={t.code}>
                  {t.code}
                  {t.name ? ` — ${t.name}` : ""}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.sitBlockWide}>
            <div className={styles.sitLabel}>
              Focus (model setting) <span className={styles.sitHint}>Need a bucket ↔ Must get a stop</span>
            </div>
            <input
              className={styles.slider}
              type="range"
              min={0}
              max={100}
              step={5}
              aria-label="Offense or defense focus"
              value={focus}
              onChange={(e) => setFocus(Number(e.target.value))}
            />
            <div className={styles.sliderEnds}>
              <span>Bucket</span>
              <span>Stop</span>
            </div>
          </div>

          <div className={styles.sitBlock}>
            <div className={styles.sitLabel}>Plays shown (display setting)</div>
            <select
              className={styles.select}
              aria-label="Plays to show"
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
            <div className={styles.sitLabel}>Recommendations (model setting)</div>
            <select
              className={styles.select}
              aria-label="Ranking method"
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
              <Chip active={ctxEndQ} label="End of period (note)" onClick={() => setCtxEndQ((v) => !v)} />
              <Chip active={ctxVsSwitching} label="Vs switching" onClick={() => setCtxVsSwitching((v) => !v)} />
            </div>
            <p className={styles.smallMuted}>End of period is a note only. Other selected conditions adjust smart rankings; baseline ignores them.</p>
          </div>

          <div className={styles.contextRight}>
            <div className={styles.miniInputs}>
              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Period (5 = OT)</div>
                <input
                  className={styles.input}
                  type="number"
                  min={1}
                  max={5}
                  aria-label="Period"
                  placeholder="1–5"
                  value={quarter ?? ""}
                  onChange={(e) => {
                    if (e.target.value === "") return setQuarter(null);
                    const nextPeriod = clamp(Math.round(Number(e.target.value)), 1, 5);
                    setQuarter(nextPeriod);
                    setTimeLeft((seconds) => (seconds === null ? null : Math.min(seconds, periodSeconds(nextPeriod))));
                  }}
                />
              </div>

              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Time left (sec)</div>
                <input
                  className={styles.input}
                  type="number"
                  min={0}
                  max={periodSeconds(quarter ?? 1)}
                  aria-label="Time left in seconds"
                  placeholder="e.g. 40"
                  value={timeLeft ?? ""}
                  onChange={(e) => setTimeLeft(e.target.value === "" ? null : clamp(Number(e.target.value), 0, periodSeconds(quarter ?? 1)))}
                />
              </div>

              <div className={styles.miniBlock}>
                <div className={styles.sitLabel}>Score (our − opp)</div>
                <input
                  className={styles.input}
                  type="number"
                  min={-50}
                  max={50}
                  aria-label="Score margin"
                  placeholder="e.g. -3"
                  value={margin ?? ""}
                  onChange={(e) => setMargin(e.target.value === "" ? null : clamp(Number(e.target.value), -50, 50))}
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
            Team and season options are unavailable. Refresh the page to try again.
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
              <span className={styles.smallMuted}>
                {selectedPlay ? (selectedPlay.source === "smart" ? "ML + context" : "Season baseline") : "Choose a play to see its diagram and counters."}
              </span>
            </div>

            {resultState.changed ? <p className={styles.warn} role="status">Inputs changed. Build Gameplan again to update the recommendations.</p> : null}
            {resultState.current && rankingNote ? <p className={styles.warn} role="status">{rankingNote}</p> : null}
            {top3.length > 0 && recommendationStyle === "smart" && !smartPlays.length && !rankingNote ? (
              <p className={styles.smallMuted} role="status">
                {loading ? "Showing the season baseline while context results load." : "No context ranking was returned. Showing the season baseline."} Score, clock and coaching conditions do not affect these baseline results.
              </p>
            ) : null}
            {top3.length && recommendationStyle === "baseline" ? (
              <p className={styles.smallMuted}>Baseline uses season matchup data. Score, period, clock and coaching conditions do not change these results.</p>
            ) : null}
            {top3.length && recommendationStyle === "smart" && smartPlays.length ? (
              <p className={styles.smallMuted}>Coaching conditions add rule-based adjustments to the matchup estimates. They are not measured situational success rates.</p>
            ) : null}
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
                          <div className={styles.quickName}>{idx === 0 ? <TopPickCall rows={activePlays} current={!loading} /> : p.playType}</div>
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
                        <div className={styles.smallMuted}>Why consider it</div>
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
        <div className={styles.right}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Shot Plan</h2>
              <span className={styles.smallMuted}>Shot types and court zones</span>
            </div>

            <div className={styles.shotPlan}>
              {loading || shotLoading ? (
                <p className={styles.empty} role="status">Finding shot priorities for this matchup…</p>
              ) : !resultState.shotsCurrent ? (
                <p className={styles.empty}>{shotError ? "Shot priorities could not load. Build the gameplan again to retry." : "Build a gameplan to see shot priorities for this matchup."}</p>
              ) : <GameplanShotAdvice shotType={selectedShotType} zone={selectedZone} />}

              <div className={styles.shotBtns}>
                <button
                  className={styles.btnSoft}
                  type="button"
                  onClick={() => setVizTab("shotHeatmap")}
                  disabled={!resultState.shotsCurrent}
                >
                  {shotLoading ? "Loading…" : "Show heatmap"}
                </button>

                <button
                  className={styles.btnSoft}
                  type="button"
                  disabled={!resultState.shotsCurrent}
                  onClick={() => {
                    if (!resultState.shotsCurrent || !resultInputs) return;
                    authenticatedDownload(getShotPlanPdfUrl({
                      season: resultInputs.season,
                      our: resultInputs.our,
                      opp: resultInputs.opp,
                      k: 5,
                      wOff: resultInputs.wOff,
                      shotType: selectedShotType || undefined,
                      zone: selectedZone || undefined,
                    })).catch(e => alert(e));
                  }}
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
                    disabled={!resultState.shotsCurrent || !shotTypeOptions.length}
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
                    disabled={!resultState.shotsCurrent || !zoneOptions.length}
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
              <span className={styles.smallMuted}>Play diagram or recorded shot locations</span>
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

                  {playVizLoading || loadedPlayVizKey !== playVizKey ? (
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
                        <GameplanDiagramCaption caption={playVizCaption} />
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

                {!resultState.shotsCurrent ? (
                  <div className={styles.empty}>Build a gameplan to fetch shot data, then open this tab.</div>
                ) : shotVizLoading || loadedShotVizKey !== shotVizKey ? (
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
                <input className={styles.input} placeholder="Name" value={roles.ballHandler} disabled={syncStatus === "loading"} onChange={(e) => setRoles({ ...roles, ballHandler: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Screener / Roller</div>
                <input className={styles.input} placeholder="Name" value={roles.screener} disabled={syncStatus === "loading"} onChange={(e) => setRoles({ ...roles, screener: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Corner Spacer</div>
                <input className={styles.input} placeholder="Name" value={roles.cornerSpacer} disabled={syncStatus === "loading"} onChange={(e) => setRoles({ ...roles, cornerSpacer: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Weak-side Cutter</div>
                <input className={styles.input} placeholder="Name" value={roles.cutter} disabled={syncStatus === "loading"} onChange={(e) => setRoles({ ...roles, cutter: e.target.value })} />
              </div>

              <div className={styles.roleRow}>
                <div className={styles.roleLabel}>Safety Outlet</div>
                <input className={styles.input} placeholder="Name" value={roles.safety} disabled={syncStatus === "loading"} onChange={(e) => setRoles({ ...roles, safety: e.target.value })} />
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

            <div className={styles.smallMuted} role="status" aria-live="polite" data-sync-status={syncStatus}>
              {SYNC_LABELS[syncStatus]}
              {syncNote ? ` (${syncNote})` : ""}
              {syncStatus === "error" ? (
                <>
                  {" "}
                  <button type="button" className={styles.chip} onClick={() => void syncNow()}>
                    Retry
                  </button>
                </>
              ) : null}
              {offerDraft && syncStatus !== "loading" ? (
                <>
                  {" "}
                  <button
                    type="button"
                    className={styles.chip}
                    onClick={() => {
                      applyState(offerDraft.state);
                      setOfferDraft(null);
                    }}
                  >
                    {offerDraft.label}
                  </button>
                </>
              ) : null}
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
                  disabled={syncStatus === "loading"}
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
        </div>
      </div>
    </div>
  );
}
