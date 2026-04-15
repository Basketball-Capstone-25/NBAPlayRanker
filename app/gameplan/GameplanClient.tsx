// app/gameplan/GameplanClient.tsx
//
// Gameplan (Client):
// - Coach-friendly “set the situation → get top options → see visuals/counters → build a plan” flow.
// - Combines Dataset1 (play-type rankings + play zones viz) + Dataset2 (shot plan + heatmap + PDF).
// - NLP is optional, but when used it now preserves and passes the full advanced context cleanly
//   through parse -> ranking -> explanation -> UI state.
// - UI stores plan/notes/roles locally (localStorage) so the demo is smooth.

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Gameplan.module.css";

import {
  baselineRank,
  contextRank,
  explainNlpRecommendations,
  fetchMetaOptions,
  fetchPlaytypeViz,
  fetchShotHeatmap,
  fetchShotPlanRank,
  getShotPlanPdfUrl,
  parseNlpPrompt,
  type AdvancedContextPayload,
  type MetaOptions,
  type NlpExplainPlay,
} from "../utils";

type TeamOption = { code: string; name?: string };

type NormalizedPlay = {
  id: string;
  playType: string;
  ppp: number | null;
  edge: number | null;
  offense: number | null;
  defense: number | null;
  why: string[];
  source: "smart" | "baseline";
  raw: any;
};

type PlanItem = { id: string; label: string };

type RoleAssignments = {
  ballHandler: string;
  screener: string;
  cornerSpacer: string;
  cutter: string;
  safety: string;
};

type ExplainItem = NlpExplainPlay;

type BuildContextSnapshot = {
  period: number;
  time_remaining: number;
  margin: number;
  after_timeout: boolean;
  late_clock: boolean;
  need3: boolean;
  protect_lead: boolean;
  end_of_quarter: boolean;
  vs_switching: boolean;
  must_stop: boolean;
  quick2: boolean;
  two_for_one: boolean;
  advanced: AdvancedContextPayload;
};

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function toNumMaybe(v: any): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function normalizeTeams(meta: MetaOptions | null): TeamOption[] {
  const teams = (meta?.teams ?? []) as any[];
  const teamNames = (meta?.teamNames ?? {}) as Record<string, string>;
  return teams
    .map((t) => String(t))
    .filter(Boolean)
    .map((code) => ({ code, name: teamNames[code] }));
}

function dedupeStrings(values: Array<string | null | undefined>) {
  const out: string[] = [];
  const seen = new Set<string>();

  for (const value of values) {
    if (typeof value !== "string") continue;
    const clean = value.trim();
    if (!clean || seen.has(clean)) continue;
    out.push(clean);
    seen.add(clean);
  }

  return out;
}

function normalizeWhy(raw: any): string[] {
  const w =
    raw?.why ??
    raw?.rationale ??
    raw?.RATIONALE ??
    raw?.explanation ??
    raw?.reasoning ??
    raw?.notes ??
    null;

  if (!w) return [];
  if (Array.isArray(w))
    return w
      .map((x) => String(x))
      .map((s) => s.trim())
      .filter(Boolean);

  if (typeof w === "string") {
    const text = w.replace(/\r/g, "");
    const lines = text.split("\n");

    const out: string[] = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      const parts = trimmed
        .split(/(?:•|·|\u2022)/g)
        .map((p) => p.trim())
        .filter(Boolean)
        .map((p) => (p.startsWith("-") ? p.replace(/^\-+\s*/, "") : p))
        .filter(Boolean);

      if (parts.length) out.push(...parts);
      else out.push(trimmed.replace(/^\-+\s*/, ""));
    }

    if (out.length <= 1 && text.includes(" - ")) {
      return text
        .split(" - ")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
    }

    return out;
  }

  return [];
}

function normalizePlay(
  raw: any,
  idx: number,
  source: "smart" | "baseline"
): NormalizedPlay {
  const playType =
    raw?.PLAY_TYPE ??
    raw?.play_type ??
    raw?.playType ??
    raw?.play ??
    raw?.name ??
    raw?.label ??
    `Play ${idx + 1}`;

  const ppp =
    toNumMaybe(raw?.PPP_CONTEXT) ??
    toNumMaybe(raw?.PPP_PRED) ??
    toNumMaybe(raw?.PPP_ML_BLEND) ??
    toNumMaybe(raw?.ppp) ??
    toNumMaybe(raw?.pred_ppp) ??
    null;

  const edge =
    toNumMaybe(raw?.DELTA_VS_BASELINE) ??
    toNumMaybe(raw?.PPP_GAP) ??
    toNumMaybe(raw?.edge) ??
    null;

  const offense =
    toNumMaybe(raw?.PPP_ML_BLEND) ??
    toNumMaybe(raw?.PPP_OFF_SHRUNK) ??
    toNumMaybe(raw?.offense) ??
    null;

  const defense =
    toNumMaybe(raw?.PPP_BASELINE) ??
    toNumMaybe(raw?.PPP_DEF_SHRUNK) ??
    toNumMaybe(raw?.defense) ??
    null;

  const why = normalizeWhy(raw);

  return {
    id: `${String(playType)}__${idx}__${source}`,
    playType: String(playType),
    ppp,
    edge,
    offense,
    defense,
    why,
    source,
    raw,
  };
}

function prettyTime(seconds: number) {
  const s = clamp(seconds, 0, 720);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function prettyPeriodLabel(period: number) {
  return period === 5 ? "OT" : `Q${period}`;
}

function gradeFromRank(rank: number) {
  if (rank === 0) return "A";
  if (rank === 1) return "B";
  return "C";
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

function coachingCuesFor(playType: string): string[] {
  const p = playType.toLowerCase();

  if (p.includes("pr") || p.includes("pick") || p.includes("roll")) {
    return [
      "Screen: be still and hit the defender’s path.",
      "Ball: get shoulder-to-hip and force 2 defenders to react.",
      "Roll: sprint to the rim with hands ready.",
    ];
  }
  if (p.includes("spot")) {
    return [
      "Be shot-ready on the catch (feet set, hands ready).",
      "Attack closeouts with 1–2 dribbles max.",
      "If help comes, make the simple kick-out.",
    ];
  }
  if (p.includes("transition")) {
    return [
      "Run wide lanes: first big to rim, second to trail.",
      "Get an early paint touch → then spray to shooters.",
      "If nothing early, flow right into a ball-screen.",
    ];
  }
  if (p.includes("isolation") || p.includes("iso")) {
    return [
      "Clear a side and hold corners (spacing is the play).",
      "Get to your first advantage move quickly (no dancing).",
      "If help comes, kick early and re-attack.",
    ];
  }
  if (p.includes("handoff") || p.includes("dh")) {
    return [
      "Sell the handoff and turn the corner tight.",
      "Big: flip and screen if the defender recovers.",
      "Weakside: stay spaced, ready for the skip pass.",
    ];
  }
  if (p.includes("post")) {
    return [
      "Seal first: high-to-low, show a clear target hand.",
      "Perimeter: cut hard when your defender turns their head.",
      "If doubled, pass out early and relocate.",
    ];
  }

  return [
    "Keep spacing (corners & slots) to open driving lanes.",
    "First read fast: rim → kick → swing.",
    "If the defense loads up, move it with one extra pass.",
  ];
}

function counterPlanFor(playType: string) {
  const base = playType.toLowerCase();
  const isPnR = base.includes("pick") || base.includes("roll") || base.includes("pr");

  return [
    {
      title: "If they switch…",
      trigger: "When their big ends up on our ball handler (or they switch everything).",
      cues: isPnR
        ? ["Re-screen quickly (flip it).", "Hit the slip early.", "Attack the mismatch with pace."]
        : ["Get into a quick re-screen.", "Cut behind help.", "Throw the skip if they load up."],
      outcome: "Goal: create a mismatch or force help.",
    },
    {
      title: "If they go under…",
      trigger: "When the on-ball defender ducks under the screen.",
      cues: isPnR
        ? ["Re-screen higher (pull-up space).", "Use a handoff into flow.", "Punish with the catch-and-shoot."]
        : ["Shorten the route for a quick shot.", "Sprint into a second action.", "Keep the ball moving."],
      outcome: "Goal: get a clean rhythm shot.",
    },
    {
      title: "If they trap/hedge…",
      trigger: "When they send two to the ball to take away the first option.",
      cues: isPnR
        ? ["Hit the short roll.", "Corners stay lifted for the skip.", "One more pass = open shot."]
        : ["Flash middle as an outlet.", "Quick swing to the weak side.", "Attack the closeout."],
      outcome: "Goal: beat the trap with spacing + quick pass.",
    },
  ];
}

function contextTagSummary(context: AdvancedContextPayload | null): string[] {
  if (!context) return [];

  const tags: string[] = [];

  if (context.need) tags.push(`Need: ${context.need}`);
  if (context.defense_style) tags.push(`Defense: ${context.defense_style}`);
  if (context.pace) tags.push(`Pace: ${context.pace}`);

  if (context.after_timeout) tags.push("After timeout");
  if (context.slob) tags.push("SLOB");
  if (context.blob) tags.push("BLOB");
  if (context.advance_ball) tags.push("Advance ball");
  if (context.late_clock) tags.push("Late clock");
  if (context.need3) tags.push("Need 3");
  if (context.protect_lead) tags.push("Protect lead");
  if (context.end_of_quarter) tags.push("End of quarter");
  if (context.vs_switching) tags.push("Vs switching");
  if (context.must_stop) tags.push("Need stop");
  if (context.quick2) tags.push("Quick 2");
  if (context.two_for_one) tags.push("2-for-1");
  if (context.hold_for_last) tags.push("Last shot");
  if (context.foul_game) tags.push("Foul game");
  if (context.no_three) tags.push("No 3");

  (context.special_situations ?? []).forEach((x) => tags.push(String(x)));
  (context.preferred_play_families ?? []).forEach((x) => tags.push(`Family: ${x}`));

  return dedupeStrings(tags).slice(0, 8);
}

function mergeAdvancedContext(
  base: AdvancedContextPayload,
  patch?: AdvancedContextPayload | null
): AdvancedContextPayload {
  if (!patch) return { ...base };

  return {
    ...base,
    ...patch,
    needs: dedupeStrings([...(base.needs ?? []), ...(patch.needs ?? [])]),
    special_situations: dedupeStrings([
      ...(base.special_situations ?? []),
      ...(patch.special_situations ?? []),
    ]),
    preferred_play_families: dedupeStrings([
      ...(base.preferred_play_families ?? []),
      ...(patch.preferred_play_families ?? []),
    ]),
    intent_tags: dedupeStrings([
      ...(((base as any).intent_tags ?? []) as string[]),
      ...((((patch as any).intent_tags ?? []) as string[]) ?? []),
    ]) as any,
  };
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
  const [parsedContext, setParsedContext] = useState<AdvancedContextPayload | null>(null);
  const [parsedContextSummary, setParsedContextSummary] = useState<string>("");

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
  const [explainOverallSummary, setExplainOverallSummary] = useState<string>("");
  const [explainContextSummary, setExplainContextSummary] = useState<string>("");
  const [explainNotes, setExplainNotes] = useState<string[]>([]);

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

  const latestBuildRef = useRef<number>(0);

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
    return `${prettyPeriodLabel(quarter)} • ${t} left • ${upDown}`;
  }, [quarter, timeLeft, margin]);

  const currentUiContext = useMemo<AdvancedContextPayload>(() => {
    const needs = dedupeStrings([
      ctxNeed3 ? "need3" : null,
      ctxProtectLead ? "safe" : null,
    ]);

    const specialSituations = dedupeStrings([
      ctxAfterTimeout ? "after_timeout" : null,
    ]);

    return {
      period: quarter,
      time_remaining: timeLeft,
      margin,
      after_timeout: ctxAfterTimeout,
      late_clock: ctxLateClock,
      need3: ctxNeed3,
      protect_lead: ctxProtectLead,
      end_of_quarter: ctxEndQ,
      vs_switching: ctxVsSwitching,
      needs,
      need: ctxNeed3 ? "need3" : ctxProtectLead ? "safe" : null,
      defense_style: ctxVsSwitching ? "switch" : null,
      special_situations: specialSituations,
      shot_clock: null,
    };
  }, [
    quarter,
    timeLeft,
    margin,
    ctxAfterTimeout,
    ctxLateClock,
    ctxNeed3,
    ctxProtectLead,
    ctxEndQ,
    ctxVsSwitching,
  ]);

  const liveAdvancedContext = useMemo<AdvancedContextPayload>(() => {
    return mergeAdvancedContext(currentUiContext, parsedContext);
  }, [currentUiContext, parsedContext]);

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

  function applyNlpContextToUi(c?: AdvancedContextPayload | null): BuildContextSnapshot {
    const nextAdvanced = mergeAdvancedContext(currentUiContext, c ?? {});

    const next: BuildContextSnapshot = {
      period: clamp(Number(nextAdvanced.period ?? quarter), 1, 5),
      time_remaining: clamp(Number(nextAdvanced.time_remaining ?? timeLeft), 0, 720),
      margin: clamp(Number(nextAdvanced.margin ?? margin), -30, 30),
      after_timeout: Boolean(nextAdvanced.after_timeout ?? ctxAfterTimeout),
      late_clock: Boolean(nextAdvanced.late_clock ?? ctxLateClock),
      need3: Boolean(nextAdvanced.need3 ?? ctxNeed3),
      protect_lead: Boolean(nextAdvanced.protect_lead ?? ctxProtectLead),
      end_of_quarter: Boolean(nextAdvanced.end_of_quarter ?? ctxEndQ),
      vs_switching: Boolean(
        nextAdvanced.vs_switching ??
          (nextAdvanced.defense_style === "switch" ? true : ctxVsSwitching)
      ),
      must_stop: Boolean(nextAdvanced.must_stop),
      quick2: Boolean(nextAdvanced.quick2),
      two_for_one: Boolean(nextAdvanced.two_for_one),
      advanced: nextAdvanced,
    };

    setQuarter(next.period);
    setTimeLeft(next.time_remaining);
    setMargin(next.margin);
    setCtxAfterTimeout(next.after_timeout);
    setCtxLateClock(next.late_clock);
    setCtxNeed3(next.need3);
    setCtxProtectLead(next.protect_lead);
    setCtxEndQ(next.end_of_quarter);
    setCtxVsSwitching(next.vs_switching);

    if (next.must_stop) setFocus(70);
    if (next.need3 || next.quick2) setFocus(25);

    setParsedContext(nextAdvanced);
    setParsedContextSummary(
      typeof nextAdvanced.context_brief === "string" ? nextAdvanced.context_brief : ""
    );

    return next;
  }

  async function runNlpParse(): Promise<BuildContextSnapshot | null> {
    if (!nlText.trim()) {
      setNlConfidence(null);
      setNlQuestions([]);
      setNlWarnings([]);
      setParsedContext(null);
      setParsedContextSummary("");
      return null;
    }

    try {
      setStatusHint("Understanding situation…");

      const res = await parseNlpPrompt({
        text: nlText,
        defaults: currentUiContext,
      });

      setNlConfidence(res.confidence ?? null);
      setNlQuestions(res.clarifying_questions ?? []);
      setNlWarnings(res.warnings ?? []);

      const snapshot = applyNlpContextToUi((res.context ?? {}) as AdvancedContextPayload);

      setStatusHint("Situation parsed ✅");
      setTimeout(() => setStatusHint(""), 900);
      return snapshot;
    } catch (e: any) {
      setStatusHint(e?.message || "Couldn’t parse the situation.");
      return null;
    }
  }

  async function buildGameplan() {
    const myBuildId = Date.now();
    latestBuildRef.current = myBuildId;

    setLoading(true);
    setStatusHint("");
    setExplainMap({});
    setExplainOverallSummary("");
    setExplainContextSummary("");
    setExplainNotes([]);

    let effectiveContext = currentUiContext;
    let effectivePeriod = quarter;
    let effectiveTimeLeft = timeLeft;
    let effectiveMargin = margin;

    try {
      if (nlText.trim()) {
        const parsed = await runNlpParse();
        if (parsed) {
          effectiveContext = parsed.advanced;
          effectivePeriod = parsed.period;
          effectiveTimeLeft = parsed.time_remaining;
          effectiveMargin = parsed.margin;
        }
      } else {
        effectiveContext = mergeAdvancedContext(currentUiContext, parsedContext);
      }

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

      let smart: NormalizedPlay[] = [];
      let smartRankingsRaw: any[] = [];
      let smartContextSummary = "";
      let smartNotes: string[] = [];
      let smartWarnings: string[] = [];

      try {
        const smartRankRes = await contextRank({
          season,
          our,
          opp,
          margin: effectiveMargin,
          period: effectivePeriod,
          timeRemaining: effectiveTimeLeft,
          k: clamp(showPlays, 1, 12),
          wOff,
          wDef,
          context: effectiveContext,
        });

        smartRankingsRaw = Array.isArray(smartRankRes.rankings)
          ? smartRankRes.rankings.map((x) => x.raw)
          : [];

        smart = smartRankingsRaw.map((x, i) => normalizePlay(x, i, "smart"));
        smartContextSummary =
          typeof (smartRankRes.context as any)?.context_brief === "string"
            ? (smartRankRes.context as any).context_brief
            : "";
        smartNotes = Array.isArray(smartRankRes.notes) ? smartRankRes.notes : [];
        smartWarnings = Array.isArray(smartRankRes.warnings) ? smartRankRes.warnings : [];

        if (latestBuildRef.current !== myBuildId) return;
        setSmartPlays(smart);
      } catch {
        smart = [];
        smartRankingsRaw = [];
        smartContextSummary = "";
        smartNotes = [];
        smartWarnings = [];
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

      try {
        const explain = await explainNlpRecommendations({
          context: effectiveContext as Record<string, any>,
          rankedContext:
            recommendationStyle === "baseline" || !smartRankingsRaw.length
              ? baseRank.map((x) => x.raw)
              : smartRankingsRaw,
          rankedBaseline: baseRank.map((x) => x.raw),
          topK: clamp(showPlays, 1, 12),
          parserWarnings: [...nlWarnings, ...smartWarnings],
          clarifyingQuestions: nlQuestions,
        });

        if (latestBuildRef.current !== myBuildId) return;

        const items = Array.isArray(explain.plays) ? explain.plays : [];

        const map: Record<string, ExplainItem> = {};
        for (const item of items) {
          const key = String(item.play_name ?? item.play_type ?? "");
          if (key) map[key] = item;
        }

        setExplainMap(map);
        setExplainOverallSummary(explain.overall_summary ?? "");
        setExplainContextSummary(
          explain.context_summary || smartContextSummary || parsedContextSummary || ""
        );
        setExplainNotes(Array.isArray(explain.notes) ? explain.notes : smartNotes);
      } catch {
        setExplainContextSummary(smartContextSummary || parsedContextSummary || "");
        setExplainNotes(smartNotes);
      }

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

    if (liveAdvancedContext.need3) {
      bullets.push("Primary: create an open 3 (corner or slot) off a paint touch.");
    } else if (liveAdvancedContext.protect_lead) {
      bullets.push("Primary: safe, high-quality shot (rim or catch-and-shoot). No risky passes.");
    } else if (liveAdvancedContext.quick2) {
      bullets.push("Primary: get a fast, efficient 2 before the defense fully loads.");
    } else {
      bullets.push("Primary: get a clean rim touch or a catch-and-shoot 3.");
    }

    if (selectedShotType) bullets.push(`Look for: ${selectedShotType}.`);
    if (selectedZone) bullets.push(`Target zone: ${selectedZone}.`);

    if (liveAdvancedContext.vs_switching) {
      bullets.push("Vs switch: slip early, then attack the mismatch with pace.");
    }
    if (liveAdvancedContext.late_clock) {
      bullets.push("Late clock: simplify — 1 action to the rim, 1 kick-out option.");
    }
    if (liveAdvancedContext.after_timeout) {
      bullets.push("After timeout: use your cleanest first action and make the first read decisive.");
    }
    if (liveAdvancedContext.two_for_one) {
      bullets.push("2-for-1: shot timing matters — do not use the whole clock.");
    }

    if (!selectedShotType && !selectedZone) {
      bullets.push("(Shot plan will auto-fill when the Dataset2 endpoint responds.)");
    }

    return bullets;
  }, [liveAdvancedContext, selectedShotType, selectedZone]);

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

  const parsedTags = useMemo(() => contextTagSummary(parsedContext), [parsedContext]);

  return (
    <div className={styles.page}>
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

        <div className={styles.panel} style={{ marginTop: 14 }}>
          <div className={styles.panelHeader}>
            <h2 className={styles.h2}>Describe the situation</h2>
            <span className={styles.smallMuted}>
              Type it like a coach: score, time, coverage, constraints, urgency.
            </span>
          </div>

          <textarea
            className={styles.textarea}
            value={nlText}
            onChange={(e) => setNlText(e.target.value)}
            placeholder="Example: 'Down 3 with 0:28 left in Q4. Need a quick 2 or a clean 3. They are switching. After timeout.'"
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
                setParsedContext(null);
                setParsedContextSummary("");
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

          {parsedContextSummary ? (
            <div className={styles.warn}>
              <div className={styles.warnSmall}>
                <b>Parsed context:</b> {parsedContextSummary}
              </div>
              {parsedTags.length ? (
                <div className={styles.warnSmall} style={{ marginTop: 6 }}>
                  {parsedTags.join(" • ")}
                </div>
              ) : null}
            </div>
          ) : null}

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
                <div className={styles.sitLabel}>Period</div>
                <input
                  className={styles.input}
                  type="number"
                  min={1}
                  max={5}
                  value={quarter}
                  onChange={(e) => setQuarter(clamp(Number(e.target.value || 4), 1, 5))}
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

      <div className={styles.grid}>
        <aside className={styles.left}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Quick Call</h2>
              <span className={styles.smallMuted}>The “what to run next” board.</span>
            </div>

            {explainOverallSummary ? (
              <div className={styles.warn} style={{ marginBottom: 12 }}>
                <div className={styles.warnSmall}>
                  <b>Gameplan summary:</b> {explainOverallSummary}
                </div>
                {explainContextSummary ? (
                  <div className={styles.warnSmall} style={{ marginTop: 6 }}>
                    <b>Context:</b> {explainContextSummary}
                  </div>
                ) : null}
              </div>
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

            {explainNotes.length ? (
              <div className={styles.warn} style={{ marginTop: 12 }}>
                {explainNotes.slice(0, 3).map((note, idx) => (
                  <div key={idx} className={styles.warnSmall}>
                    {note}
                  </div>
                ))}
              </div>
            ) : null}
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

        <main className={styles.right}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.h2}>Shot Plan</h2>
              <span className={styles.smallMuted}>Simple “what shot we want” guidance.</span>
            </div>

            <div className={styles.shotPlan}>
              <div className={styles.shotHeadline}>
                {liveAdvancedContext.need3 ? "We want an open 3…" : "We want a high-quality shot…"}
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

                <a
                  className={styles.btnSoft}
                  href={getShotPlanPdfUrl({
                    season,
                    our,
                    opp,
                    k: 5,
                    wOff,
                    shotType: selectedShotType || undefined,
                    zone: selectedZone || undefined,
                  })}
                  target="_blank"
                  rel="noreferrer"
                >
                  Export PDF
                </a>
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
                      <span className={styles.detailKey}>Context engine</span>
                      <span className={styles.detailVal}>
                        Full NLP context is preserved through ranking and explanation.
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
