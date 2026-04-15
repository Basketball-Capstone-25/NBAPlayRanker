// app/utils.ts
//
// TypeScript API helpers for the PSPI.
//
// This updated version keeps all existing Dataset1 / Dataset2 helpers,
// and adds richer NLP + contextual ranking support for Gameplan.
//
// New capabilities:
// - Advanced context payload typing for expanded NLP fields
// - POST-first context ranking helper with GET fallback
// - NLP parse / explain helpers
// - Backward compatibility for existing pages

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export const FALLBACK_SEASONS = [
  "2019-20",
  "2020-21",
  "2021-22",
  "2022-23",
  "2023-24",
  "2024-25",
];

export const FALLBACK_TEAMS = [
  "ATL","BKN","BOS","CHA","CHI","CLE",
  "DAL","DEN","DET","GSW","HOU","IND",
  "LAC","LAL","MEM","MIA","MIL","MIN",
  "NOP","NYK","OKC","ORL","PHI","PHX",
  "POR","SAC","SAS","TOR","UTA","WAS",
];

// ---------------------------
// Small fetch helpers
// ---------------------------

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-store" });
  const text = await res.text();

  if (!res.ok) {
    throw new ApiError(res.status, url, text);
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(res.status, url, text);
  }
}

class ApiError extends Error {
  status: number;
  url: string;
  body: string;

  constructor(status: number, url: string, body: string) {
    super(`API error ${status}: ${body}`);
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

async function fetchJsonWithStatus<T>(url: string): Promise<T> {
  return await fetchJson<T>(url);
}

async function postJsonWithStatus<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();

  if (!res.ok) {
    throw new ApiError(res.status, url, text);
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(res.status, url, text);
  }
}

async function tryJsonCandidates<T>(
  urls: string[],
  opts?: {
    keepTryingOnClientError?: boolean;
  }
): Promise<{ data: T; usedUrl: string }> {
  let lastErr: unknown = null;

  for (const url of urls) {
    try {
      const data = await fetchJsonWithStatus<T>(url);
      return { data, usedUrl: url };
    } catch (e: any) {
      lastErr = e;

      if (e?.status === 404) continue;

      if (opts?.keepTryingOnClientError && (e?.status === 400 || e?.status === 422)) {
        continue;
      }

      break;
    }
  }

  const attempted = urls.map((u) => `- ${u}`).join("\n");
  const msg = (lastErr as any)?.message ?? "Unknown error";
  throw new Error(`Request failed.\nTried:\n${attempted}\n\nLast error: ${msg}`);
}

async function tryPostJsonCandidates<T>(
  requests: Array<{ url: string; body: unknown }>,
  opts?: {
    keepTryingOnClientError?: boolean;
  }
): Promise<{ data: T; usedUrl: string; usedBody: unknown }> {
  let lastErr: unknown = null;

  for (const req of requests) {
    try {
      const data = await postJsonWithStatus<T>(req.url, req.body);
      return { data, usedUrl: req.url, usedBody: req.body };
    } catch (e: any) {
      lastErr = e;

      if (e?.status === 404) continue;

      if (opts?.keepTryingOnClientError && (e?.status === 400 || e?.status === 422)) {
        continue;
      }

      break;
    }
  }

  const attempted = requests.map((r) => `- POST ${r.url}`).join("\n");
  const msg = (lastErr as any)?.message ?? "Unknown error";
  throw new Error(`Request failed.\nTried:\n${attempted}\n\nLast error: ${msg}`);
}

// ---------------------------
// Types (Dataset1)
// ---------------------------

export type MetaOptions = {
  seasons: string[];
  teams: string[];
  teamNames?: Record<string, string>;
  playTypes?: string[];
  sides?: string[];
  hasMlPredictions?: boolean;
  _fallback?: boolean;
};

export type TeamPlaytypesPreviewResponse = {
  season: string;
  total_rows: number;
  returned_rows: number;
  rows: Record<string, any>[];
};

export type BaselineRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  w_off: number;
  w_def: number;
  rankings: Record<string, any>[];
};

export type ContextRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  margin: number;
  period: number;
  time_remaining_period_sec: number;
  w_off: number;
  w_def: number;
  rankings: Record<string, any>[];
  context?: Record<string, any>;
  notes?: string[];
  warnings?: string[];
  parser_warnings?: string[];
  clarifying_questions?: string[];
};

export type ModelMetricsResponse = {
  n_splits: number;
  metrics: Array<{
    model: string;
    RMSE_mean: number;
    RMSE_std: number;
    MAE_mean: number;
    MAE_std: number;
    R2_mean: number;
    R2_std: number;
  }>;
  rf_vs_baseline_t: number | null;
  rf_vs_baseline_p: number | null;
};

export type MlAnalysisResponse = {
  dataset: any;
  eda: any;
  correlations: { labels: string[]; matrix: number[][] };
  target_feature_corr: Array<{ feature: string; corr: number; abs: number }>;
  feature_selection: any;
  model_selection: any;
};

// ---------------------------
// Types (Dataset2 / Shots Intelligence + Viz)
// ---------------------------

export type ShotPlanRankResponse = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  w_off: number;
  w_def: number;
  best_shooter?: any;
  top_shot_types: Record<string, any>[];
  top_zones: Record<string, any>[];
  top_pairs?: Record<string, any>[];
  metadata?: any;
  notes?: string[];
  _endpoint_used?: string;
};

export type ShotHeatmapResponse = {
  image_base64: string;
  caption?: string;
  season?: string;
  team?: string;
  opp?: string;
  shot_type?: string | null;
  zone?: string | null;
  max_shots?: number;
  _endpoint_used?: string;
};

export type ShotModelMetricsResponse = {
  n_splits: number;
  metrics: Array<{
    model: string;
    RMSE_mean: number;
    RMSE_std: number;
    MAE_mean: number;
    MAE_std: number;
    R2_mean: number;
    R2_std: number;
  }>;
};

export type ShotMlAnalysisResponse = {
  dataset: any;
  eda: any;
  correlations: { labels: string[]; matrix: number[][] };
  target_feature_corr: Array<{ feature: string; corr: number; abs: number }>;
  feature_selection: any;
  model_selection: any;
};

export type PbpMetaOptions = {
  seasons: string[];
  teams: string[];
  shotTypes?: string[];
  zones?: string[];
};

export type PbpShotsPreviewResponse = {
  season: string;
  team: string;
  opp?: string | null;
  shot_type?: string | null;
  zone?: string | null;
  total_rows: number;
  returned_rows: number;
  columns: string[];
  rows: Record<string, any>[];
  _endpoint_used?: string;
};

// ---------------------------
// Types (NLP / Gameplan)
// ---------------------------

export type AdvancedContextPayload = {
  period?: number | null;
  margin?: number | null;
  timeRemaining?: number | null;
  time_remaining?: number | null;
  shotClock?: number | null;
  shot_clock?: number | null;

  need?: string | null;
  needs?: string[];
  defenseStyle?: string | null;
  defense_style?: string | null;
  pace?: string | null;

  specialSituations?: string[];
  special_situations?: string[];
  preferredPlayFamilies?: string[];
  preferred_play_families?: string[];

  afterTimeout?: boolean;
  after_timeout?: boolean;
  slob?: boolean;
  blob?: boolean;
  advanceBall?: boolean;
  advance_ball?: boolean;

  lateClock?: boolean;
  late_clock?: boolean;
  need3?: boolean;
  protectLead?: boolean;
  protect_lead?: boolean;
  endOfQuarter?: boolean;
  end_of_quarter?: boolean;
  vsSwitching?: boolean;
  vs_switching?: boolean;
  mustStop?: boolean;
  must_stop?: boolean;
  quick2?: boolean;
  twoForOne?: boolean;
  two_for_one?: boolean;
  holdForLast?: boolean;
  hold_for_last?: boolean;
  foulGame?: boolean;
  foul_game?: boolean;
  noThree?: boolean;
  no_three?: boolean;
  mustScore?: boolean;
  must_score?: boolean;
  safe?: boolean;

  offenseBias?: number | null;
  offense_bias?: number | null;
  defenseBias?: number | null;
  defense_bias?: number | null;

  intentTags?: string[];
  intent_tags?: string[];

  contextBrief?: string | null;
  context_brief?: string | null;
  objectiveSummary?: string | null;
  objective_summary?: string | null;

  parserVersion?: string | null;
  parser_version?: string | null;
  rawText?: string | null;
  raw_text?: string | null;
  textNormalized?: string | null;
  text_normalized?: string | null;
};

export type ContextRankOpts = {
  season: string;
  our: string;
  opp: string;
  margin?: number;
  period?: number;
  timeRemaining?: number;
  shotClock?: number;
  k?: number;
  wOff?: number;
  wDef?: number;
  context?: AdvancedContextPayload;
};

export type ContextRankRow = {
  playType: string;
  finalPPP: number;
  mlPPP: number;
  baselinePPP: number;
  deltaPPP: number;
  contextLabel: string;
  rationale: string;
  contextAdj?: number;
  bonusQuick?: number;
  bonusScore?: number;
  penaltyProtect?: number;
  raw: Record<string, any>;
};

export type ContextRankResult = {
  season: string;
  our_team: string;
  opp_team: string;
  k: number;
  margin: number;
  period: number;
  time_remaining_period_sec: number;
  w_off: number;
  w_def: number;
  rankings: ContextRankRow[];
  context?: Record<string, any>;
  notes?: string[];
  warnings?: string[];
  parser_warnings?: string[];
  clarifying_questions?: string[];
  _endpoint_used?: string;
  _method_used?: "GET" | "POST";
};

export type NlpParseResponse = {
  raw_text?: string;
  context: Record<string, any>;
  confidence?: number;
  clarifying_questions?: string[];
  matches?: Record<string, string>;
  warnings?: string[];
  parser_version?: string;
  _endpoint_used?: string;
};

export type NlpExplainPlay = {
  play_name: string;
  play_type: string;
  rank: number;
  summary: string;
  evidence: string[];
  caution?: string | null;
  matched_context?: string[];
  metrics_used?: Record<string, any>;
};

export type NlpExplainResponse = {
  context_summary: string;
  overall_summary: string;
  plays: NlpExplainPlay[];
  notes?: string[];
  parser_warnings?: string[];
  clarifying_questions?: string[];
  explainer_version?: string;
  _endpoint_used?: string;
};

// ---------------------------
// Helpers to keep UI from crashing
// ---------------------------

function normalizeFeatureSelection(fs: any): any {
  const out = fs && typeof fs === "object" ? { ...fs } : {};

  const cfRaw = out.correlation_filter && typeof out.correlation_filter === "object"
    ? out.correlation_filter
    : {};

  out.correlation_filter = {
    threshold: cfRaw.threshold ?? null,
    kept: Array.isArray(cfRaw.kept) ? cfRaw.kept : [],
    dropped: Array.isArray(cfRaw.dropped)
      ? cfRaw.dropped
      : Array.isArray(cfRaw.removed)
        ? cfRaw.removed
        : [],
  };

  return out;
}

function unwrapMaybeCachedPayload(raw: any): any {
  if (raw && typeof raw === "object" && raw.payload && typeof raw.payload === "object") {
    return raw.payload;
  }
  return raw;
}

function normalizeAnalysisResponse<T extends { feature_selection?: any }>(raw: any): T {
  const base = unwrapMaybeCachedPayload(raw);
  const obj = base && typeof base === "object" ? { ...base } : {};
  obj.feature_selection = normalizeFeatureSelection(obj.feature_selection);
  return obj as T;
}

// ---------------------------
// Context payload normalization
// ---------------------------

function dedupeStrings(values: Array<string | null | undefined>): string[] {
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

function toSnakeCaseContext(context?: AdvancedContextPayload): Record<string, any> {
  const c = context ?? {};

  const timeRemaining =
    c.time_remaining ??
    c.timeRemaining ??
    null;

  const shotClock =
    c.shot_clock ??
    c.shotClock ??
    null;

  const defenseStyle =
    c.defense_style ??
    c.defenseStyle ??
    null;

  const specialSituations = Array.isArray(c.special_situations)
    ? c.special_situations
    : Array.isArray(c.specialSituations)
      ? c.specialSituations
      : [];

  const preferredPlayFamilies = Array.isArray(c.preferred_play_families)
    ? c.preferred_play_families
    : Array.isArray(c.preferredPlayFamilies)
      ? c.preferredPlayFamilies
      : [];

  const intentTags = Array.isArray(c.intent_tags)
    ? c.intent_tags
    : Array.isArray(c.intentTags)
      ? c.intentTags
      : [];

  return {
    period: c.period ?? null,
    margin: c.margin ?? null,
    time_remaining: timeRemaining,
    shot_clock: shotClock,

    need: c.need ?? null,
    needs: Array.isArray(c.needs) ? c.needs : [],

    defense_style: defenseStyle,
    pace: c.pace ?? null,

    special_situations: specialSituations,
    preferred_play_families: preferredPlayFamilies,

    after_timeout: c.after_timeout ?? c.afterTimeout ?? undefined,
    slob: c.slob ?? undefined,
    blob: c.blob ?? undefined,
    advance_ball: c.advance_ball ?? c.advanceBall ?? undefined,

    late_clock: c.late_clock ?? c.lateClock ?? undefined,
    need3: c.need3 ?? undefined,
    protect_lead: c.protect_lead ?? c.protectLead ?? undefined,
    end_of_quarter: c.end_of_quarter ?? c.endOfQuarter ?? undefined,
    vs_switching: c.vs_switching ?? c.vsSwitching ?? undefined,
    must_stop: c.must_stop ?? c.mustStop ?? undefined,
    quick2: c.quick2 ?? undefined,
    two_for_one: c.two_for_one ?? c.twoForOne ?? undefined,
    hold_for_last: c.hold_for_last ?? c.holdForLast ?? undefined,
    foul_game: c.foul_game ?? c.foulGame ?? undefined,
    no_three: c.no_three ?? c.noThree ?? undefined,
    must_score: c.must_score ?? c.mustScore ?? undefined,
    safe: c.safe ?? undefined,

    offense_bias: c.offense_bias ?? c.offenseBias ?? undefined,
    defense_bias: c.defense_bias ?? c.defenseBias ?? undefined,

    intent_tags: intentTags,

    context_brief: c.context_brief ?? c.contextBrief ?? undefined,
    objective_summary: c.objective_summary ?? c.objectiveSummary ?? undefined,

    parser_version: c.parser_version ?? c.parserVersion ?? undefined,
    raw_text: c.raw_text ?? c.rawText ?? undefined,
    text_normalized: c.text_normalized ?? c.textNormalized ?? undefined,
  };
}

function compactObject<T extends Record<string, any>>(obj: T): T {
  const out: Record<string, any> = {};

  Object.entries(obj).forEach(([key, value]) => {
    if (value === undefined) return;
    if (Array.isArray(value) && value.length === 0) return;
    out[key] = value;
  });

  return out as T;
}

function buildExpandedContextFromOpts(opts: ContextRankOpts): Record<string, any> {
  const normalized = toSnakeCaseContext(opts.context);

  const merged = compactObject({
    ...normalized,
    period: normalized.period ?? opts.period ?? null,
    margin: normalized.margin ?? opts.margin ?? null,
    time_remaining: normalized.time_remaining ?? opts.timeRemaining ?? null,
    shot_clock: normalized.shot_clock ?? opts.shotClock ?? undefined,
  });

  merged.needs = dedupeStrings(Array.isArray(merged.needs) ? merged.needs : []);
  merged.special_situations = dedupeStrings(
    Array.isArray(merged.special_situations) ? merged.special_situations : []
  );
  merged.preferred_play_families = dedupeStrings(
    Array.isArray(merged.preferred_play_families) ? merged.preferred_play_families : []
  );
  merged.intent_tags = dedupeStrings(
    Array.isArray(merged.intent_tags) ? merged.intent_tags : []
  );

  return merged;
}

function normalizeContextRankResponse(
  raw: any,
  fallback: {
    season: string;
    our: string;
    opp: string;
    k: number;
    wOff: number;
    wDef: number;
    context: Record<string, any>;
  },
  endpointUsed: string,
  methodUsed: "GET" | "POST"
): ContextRankResult {
  const data = unwrapMaybeCachedPayload(raw);
  const rows = Array.isArray(data?.rankings) ? data.rankings : [];

  const normalizedRows: ContextRankRow[] = rows.map((r: Record<string, any>) => ({
    playType:
      r.PLAY_TYPE ??
      r.play_type ??
      r.playType ??
      r.name ??
      "Unknown Play",
    finalPPP: Number(r.PPP_CONTEXT ?? r.finalPPP ?? r.context_ppp ?? 0),
    mlPPP: Number(r.PPP_ML_BLEND ?? r.mlPPP ?? r.ml_ppp ?? 0),
    baselinePPP: Number(r.PPP_BASELINE ?? r.baselinePPP ?? r.baseline_ppp ?? r.PPP_PRED ?? 0),
    deltaPPP: Number(r.DELTA_VS_BASELINE ?? r.deltaPPP ?? r.delta_vs_baseline ?? 0),
    contextLabel: String(r.CONTEXT_LABEL ?? r.contextLabel ?? r.context_label ?? ""),
    rationale: String(r.RATIONALE ?? r.rationale ?? ""),
    contextAdj:
      r.CONTEXT_ADJ != null ? Number(r.CONTEXT_ADJ) :
      r.context_adj != null ? Number(r.context_adj) :
      undefined,
    bonusQuick:
      r.BONUS_QUICK != null ? Number(r.BONUS_QUICK) :
      r.bonus_quick != null ? Number(r.bonus_quick) :
      undefined,
    bonusScore:
      r.BONUS_SCORE != null ? Number(r.BONUS_SCORE) :
      r.bonus_score != null ? Number(r.bonus_score) :
      undefined,
    penaltyProtect:
      r.PENALTY_PROTECT != null ? Number(r.PENALTY_PROTECT) :
      r.penalty_protect != null ? Number(r.penalty_protect) :
      undefined,
    raw: r,
  }));

  const margin =
    Number(data?.margin ?? fallback.context.margin ?? 0);

  const period =
    Number(data?.period ?? fallback.context.period ?? 4);

  const timeRemaining =
    Number(
      data?.time_remaining_period_sec ??
      data?.time_remaining ??
      fallback.context.time_remaining ??
      0
    );

  return {
    season: String(data?.season ?? fallback.season),
    our_team: String(data?.our_team ?? data?.our ?? fallback.our),
    opp_team: String(data?.opp_team ?? data?.opp ?? fallback.opp),
    k: Number(data?.k ?? fallback.k),
    margin,
    period,
    time_remaining_period_sec: timeRemaining,
    w_off: Number(data?.w_off ?? fallback.wOff),
    w_def: Number(data?.w_def ?? fallback.wDef),
    rankings: normalizedRows,
    context: data?.context ?? fallback.context,
    notes: Array.isArray(data?.notes) ? data.notes : [],
    warnings: Array.isArray(data?.warnings) ? data.warnings : [],
    parser_warnings: Array.isArray(data?.parser_warnings) ? data.parser_warnings : [],
    clarifying_questions: Array.isArray(data?.clarifying_questions) ? data.clarifying_questions : [],
    _endpoint_used: endpointUsed,
    _method_used: methodUsed,
  };
}

// ---------------------------
// Meta (Dataset1)
// ---------------------------

export async function fetchMetaOptions(): Promise<MetaOptions> {
  try {
    return await fetchJson<MetaOptions>(`${API_BASE}/meta/options`);
  } catch {
    return {
      seasons: FALLBACK_SEASONS,
      teams: FALLBACK_TEAMS,
      teamNames: {},
      playTypes: [],
      sides: ["offense", "defense"],
      hasMlPredictions: false,
      _fallback: true,
    };
  }
}

// ---------------------------
// Dataset2 (PBP) Meta
// ---------------------------

export async function fetchPbpMetaOptions(): Promise<PbpMetaOptions> {
  try {
    return await fetchJson<PbpMetaOptions>(`${API_BASE}/pbp/meta/options`);
  } catch {
    const m = await fetchMetaOptions();
    return {
      seasons: m.seasons ?? FALLBACK_SEASONS,
      teams: m.teams ?? FALLBACK_TEAMS,
      shotTypes: [],
      zones: [],
    };
  }
}

export async function fetchPipelineInfo(): Promise<any> {
  return await fetchJson(`${API_BASE}/meta/pipeline`);
}

export async function fetchBaselineInfo(): Promise<any> {
  return await fetchJson(`${API_BASE}/meta/baseline-formula`);
}

// ---------------------------
// Data Explorer (Dataset1)
// ---------------------------

export async function fetchTeamPlaytypesPreview(opts: {
  season: string;
  team?: string;
  side?: string;
  playType?: string;
  minPoss?: number;
  limit?: number;
}): Promise<TeamPlaytypesPreviewResponse> {
  const { season, team, side, playType, minPoss = 0, limit = 200 } = opts;

  const params = new URLSearchParams();
  params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  params.set("min_poss", String(minPoss));
  params.set("limit", String(limit));

  return await fetchJson<TeamPlaytypesPreviewResponse>(
    `${API_BASE}/data/team-playtypes?${params.toString()}`
  );
}

export function getTeamPlaytypesCsvUrl(opts: {
  season: string;
  team?: string;
  side?: string;
  playType?: string;
  minPoss?: number;
  limit?: number;
}): string {
  const { season, team, side, playType, minPoss = 0 } = opts;

  const params = new URLSearchParams();
  params.set("season", season);
  if (team) params.set("team", team);
  if (side) params.set("side", side);
  if (playType) params.set("play_type", playType);
  params.set("min_poss", String(minPoss));

  return `${API_BASE}/data/team-playtypes.csv?${params.toString()}`;
}

// ---------------------------
// Baseline ranking (Dataset1)
// ---------------------------

export async function baselineRank(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  wDef?: number;
}): Promise<
  Array<{
    playType: string;
    pppPred: number;
    pppOff: number;
    pppDef: number;
    pppGap: number;
    rationale: string;
    raw: Record<string, any>;
  }>
> {
  const { season, our, opp, k = 5, wOff = 0.7, wDef = 0.3 } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(wDef),
  });

  const data = await fetchJson<BaselineRankResponse>(
    `${API_BASE}/rank-plays/baseline?${params.toString()}`
  );

  const rankings = Array.isArray(data.rankings) ? data.rankings : [];

  return rankings.map((r) => ({
    playType: r.PLAY_TYPE,
    pppPred: Number(r.PPP_PRED),
    pppOff: Number(r.PPP_OFF_SHRUNK),
    pppDef: Number(r.PPP_DEF_SHRUNK),
    pppGap: Number(r.PPP_GAP),
    rationale: r.RATIONALE || "",
    raw: r,
  }));
}

export function getBaselineCsvUrl(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  wDef?: number;
}): string {
  const { season, our, opp, k = 5, wOff = 0.7 } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
    w_def: String(opts.wDef ?? 0.3),
  });

  return `${API_BASE}/rank-plays/baseline.csv?${params.toString()}`;
}

// ---------------------------
// Context + ML ranking (Dataset1)
// ---------------------------

export async function contextRank(opts: ContextRankOpts): Promise<ContextRankResult> {
  const {
    season,
    our,
    opp,
    k = 5,
    wOff = 0.7,
    wDef = opts.wDef ?? (1 - wOff),
  } = opts;

  const contextPayload = buildExpandedContextFromOpts(opts);

  const postBodies = [
    {
      season,
      our,
      opp,
      k,
      w_off: wOff,
      w_def: wDef,
      context: contextPayload,
    },
    {
      season,
      our_team: our,
      opp_team: opp,
      k,
      w_off: wOff,
      w_def: wDef,
      context: contextPayload,
    },
    {
      season,
      our,
      opp,
      k,
      w_off: wOff,
      w_def: wDef,
      ...contextPayload,
    },
  ];

  const postCandidates = [
    { url: `${API_BASE}/rank-plays/context-ml`, body: postBodies[0] },
    { url: `${API_BASE}/rank-plays/context-ml`, body: postBodies[1] },
    { url: `${API_BASE}/rank-plays/context-ml`, body: postBodies[2] },
  ];

  try {
    const { data, usedUrl } = await tryPostJsonCandidates<any>(postCandidates, {
      keepTryingOnClientError: true,
    });

    return normalizeContextRankResponse(
      data,
      {
        season,
        our,
        opp,
        k,
        wOff,
        wDef,
        context: contextPayload,
      },
      usedUrl,
      "POST"
    );
  } catch {
    const margin =
      contextPayload.margin ??
      opts.margin ??
      0;

    const period =
      contextPayload.period ??
      opts.period ??
      4;

    const timeRemaining =
      contextPayload.time_remaining ??
      opts.timeRemaining ??
      0;

    const params = new URLSearchParams({
      season,
      our,
      opp,
      margin: String(margin),
      period: String(period),
      time_remaining: String(timeRemaining),
      k: String(k),
      w_off: String(wOff),
    });

    if (opts.wDef != null) {
      params.set("w_def", String(wDef));
    }

    if (contextPayload.shot_clock != null) {
      params.set("shot_clock", String(contextPayload.shot_clock));
    }

    const getCandidates = [
      `${API_BASE}/rank-plays/context-ml?${params.toString()}`,
    ];

    const { data, usedUrl } = await tryJsonCandidates<any>(getCandidates, {
      keepTryingOnClientError: true,
    });

    return normalizeContextRankResponse(
      data,
      {
        season,
        our,
        opp,
        k,
        wOff,
        wDef,
        context: contextPayload,
      },
      usedUrl,
      "GET"
    );
  }
}

export async function fetchModelMetrics(nSplits = 5): Promise<ModelMetricsResponse> {
  const params = new URLSearchParams({ n_splits: String(nSplits) });
  return await fetchJson<ModelMetricsResponse>(
    `${API_BASE}/metrics/baseline-vs-ml?${params.toString()}`
  );
}

// ---------------------------
// Statistical Analysis (Dataset1)
// ---------------------------

export async function fetchMlAnalysis(opts?: {
  nSplits?: number;
  minPoss?: number;
  refresh?: boolean;
}): Promise<MlAnalysisResponse> {
  const nSplits = opts?.nSplits ?? 5;
  const minPoss = opts?.minPoss ?? 25;

  const params = new URLSearchParams();
  params.set("n_splits", String(nSplits));
  params.set("min_poss", String(minPoss));
  if (opts?.refresh) params.set("refresh", "true");

  const raw = await fetchJson<MlAnalysisResponse>(
    `${API_BASE}/analysis/ml?${params.toString()}`
  );

  return normalizeAnalysisResponse<MlAnalysisResponse>(raw);
}

// ---------------------------
// SportyPy Visualization (Dataset1)
// ---------------------------

export async function fetchPlaytypeViz(opts: {
  season: string;
  our: string;
  opp: string;
  playType: string;
  wOff: number;
}) {
  const params = new URLSearchParams({
    season: opts.season,
    our: opts.our,
    opp: opts.opp,
    play_type: opts.playType,
    w_off: String(opts.wOff),
  });

  return await fetchJson<{ caption: string; image_base64: string }>(
    `${API_BASE}/viz/playtype-zones?${params.toString()}`
  );
}

// ---------------------------
// NLP / Gameplan helpers
// ---------------------------

export async function parseNlpPrompt(opts: {
  text: string;
  defaults?: AdvancedContextPayload;
}): Promise<NlpParseResponse> {
  const bodyA = {
    text: opts.text,
    defaults: compactObject(toSnakeCaseContext(opts.defaults)),
  };

  const bodyB = {
    prompt: opts.text,
    defaults: compactObject(toSnakeCaseContext(opts.defaults)),
  };

  const candidates = [
    { url: `${API_BASE}/nlp/parse`, body: bodyA },
    { url: `${API_BASE}/nlp/parse`, body: bodyB },
  ];

  const { data, usedUrl } = await tryPostJsonCandidates<any>(candidates, {
    keepTryingOnClientError: true,
  });

  const base = unwrapMaybeCachedPayload(data);

  return {
    raw_text: base?.raw_text ?? opts.text,
    context: base?.context ?? {},
    confidence: base?.confidence,
    clarifying_questions: Array.isArray(base?.clarifying_questions) ? base.clarifying_questions : [],
    matches: base?.matches ?? {},
    warnings: Array.isArray(base?.warnings) ? base.warnings : [],
    parser_version: base?.parser_version,
    _endpoint_used: usedUrl,
  };
}

export async function explainNlpRecommendations(opts: {
  context: Record<string, any>;
  rankedContext: any;
  rankedBaseline?: any;
  topK?: number;
  parserWarnings?: string[];
  clarifyingQuestions?: string[];
}): Promise<NlpExplainResponse> {
  const bodyA = {
    context: opts.context,
    ranked_context: opts.rankedContext,
    ranked_baseline: opts.rankedBaseline,
    top_k: opts.topK ?? 5,
    parser_warnings: opts.parserWarnings ?? [],
    clarifying_questions: opts.clarifyingQuestions ?? [],
  };

  const bodyB = {
    context: opts.context,
    rankings: opts.rankedContext,
    baseline: opts.rankedBaseline,
    k: opts.topK ?? 5,
    parser_warnings: opts.parserWarnings ?? [],
    clarifying_questions: opts.clarifyingQuestions ?? [],
  };

  const candidates = [
    { url: `${API_BASE}/nlp/explain`, body: bodyA },
    { url: `${API_BASE}/nlp/explain`, body: bodyB },
  ];

  const { data, usedUrl } = await tryPostJsonCandidates<any>(candidates, {
    keepTryingOnClientError: true,
  });

  const base = unwrapMaybeCachedPayload(data);

  return {
    context_summary: String(base?.context_summary ?? ""),
    overall_summary: String(base?.overall_summary ?? ""),
    plays: Array.isArray(base?.plays) ? base.plays : [],
    notes: Array.isArray(base?.notes) ? base.notes : [],
    parser_warnings: Array.isArray(base?.parser_warnings) ? base.parser_warnings : [],
    clarifying_questions: Array.isArray(base?.clarifying_questions) ? base.clarifying_questions : [],
    explainer_version: base?.explainer_version,
    _endpoint_used: usedUrl,
  };
}

// ---------------------------
// Shot Intelligence (Dataset2)
// ---------------------------

export async function fetchShotPlanRank(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
}): Promise<ShotPlanRankResponse> {
  const { season, our, opp, k = 5, wOff = 0.7 } = opts;

  const pA = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
  });

  const pB = new URLSearchParams({
    season,
    our_team: our,
    opp_team: opp,
    k: String(k),
    w_off: String(wOff),
  });

  const candidates = [
    `${API_BASE}/pbp/shotplan/rank?${pA.toString()}`,
    `${API_BASE}/pbp/shotplan?${pA.toString()}`,
    `${API_BASE}/pbp/shotplan/rank?${pB.toString()}`,
    `${API_BASE}/pbp/shotplan?${pB.toString()}`,
    `${API_BASE}/shotplan/rank?${pA.toString()}`,
    `${API_BASE}/shotplan?${pA.toString()}`,
    `${API_BASE}/shotplan/rank?${pB.toString()}`,
    `${API_BASE}/shotplan?${pB.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<ShotPlanRankResponse>(candidates, {
    keepTryingOnClientError: true,
  });

  const base = unwrapMaybeCachedPayload(data);

  const safe: ShotPlanRankResponse = {
    season: (base as any)?.season ?? season,
    our_team: (base as any)?.our_team ?? (base as any)?.our ?? our,
    opp_team: (base as any)?.opp_team ?? (base as any)?.opp ?? opp,
    k: (base as any)?.k ?? k,
    w_off: (base as any)?.w_off ?? (base as any)?.wOff ?? wOff,
    w_def: (base as any)?.w_def ?? (base as any)?.wDef ?? (1 - wOff),

    top_shot_types: Array.isArray((base as any)?.top_shot_types)
      ? (base as any).top_shot_types
      : [],
    top_zones: Array.isArray((base as any)?.top_zones)
      ? (base as any).top_zones
      : [],
    top_pairs: Array.isArray((base as any)?.top_pairs)
      ? (base as any).top_pairs
      : undefined,

    best_shooter: (base as any)?.best_shooter,
    metadata: (base as any)?.metadata,
    notes: Array.isArray((base as any)?.notes) ? (base as any).notes : undefined,
    _endpoint_used: usedUrl,
  };

  return safe;
}

export async function fetchShotHeatmap(opts: {
  season: string;
  team?: string;
  our?: string;
  opp: string;
  shotType?: string;
  zone?: string;
  maxShots?: number;
}): Promise<ShotHeatmapResponse> {
  const team = opts.team ?? opts.our;
  if (!team) {
    throw new Error("fetchShotHeatmap requires either `team` or `our`.");
  }

  const pPbpOur = new URLSearchParams({
    season: opts.season,
    our: team,
    opp: opts.opp,
  });
  if (opts.shotType) pPbpOur.set("shot_type", opts.shotType);
  if (opts.zone) pPbpOur.set("zone", opts.zone);
  if (opts.maxShots != null) pPbpOur.set("max_shots", String(opts.maxShots));

  const pPbpTeam = new URLSearchParams({
    season: opts.season,
    team,
    opp: opts.opp,
  });
  if (opts.shotType) pPbpTeam.set("shot_type", opts.shotType);
  if (opts.zone) pPbpTeam.set("zone", opts.zone);
  if (opts.maxShots != null) pPbpTeam.set("max_shots", String(opts.maxShots));

  const pRoot = new URLSearchParams({
    season: opts.season,
    our: team,
    opp: opts.opp,
  });
  if (opts.shotType) pRoot.set("shot_type", opts.shotType);
  if (opts.zone) pRoot.set("zone", opts.zone);
  if (opts.maxShots != null) pRoot.set("max_shots", String(opts.maxShots));

  const candidates = [
    `${API_BASE}/pbp/viz/shot-heatmap?${pPbpOur.toString()}`,
    `${API_BASE}/pbp/viz/shot-heatmap?${pPbpTeam.toString()}`,
    `${API_BASE}/pbp/viz/heatmap?${pPbpTeam.toString()}`,
    `${API_BASE}/viz/shot-heatmap?${pRoot.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<ShotHeatmapResponse>(candidates, {
    keepTryingOnClientError: true,
  });

  const base = unwrapMaybeCachedPayload(data);

  if (!(base as any)?.image_base64) {
    throw new Error(`Heatmap response missing image_base64. Endpoint used: ${usedUrl}`);
  }

  return {
    ...base,
    caption:
      (base as any)?.caption ??
      `Shot Heatmap • ${team} vs ${opts.opp} • ${opts.season}`,
    season: (base as any)?.season ?? opts.season,
    team: (base as any)?.team ?? team,
    opp: (base as any)?.opp ?? opts.opp,
    shot_type: (base as any)?.shot_type ?? opts.shotType ?? null,
    zone: (base as any)?.zone ?? opts.zone ?? null,
    max_shots: (base as any)?.max_shots ?? opts.maxShots,
    _endpoint_used: usedUrl,
  };
}

export function getShotPlanPdfUrl(opts: {
  season: string;
  our: string;
  opp: string;
  k?: number;
  wOff?: number;
  shotType?: string;
  zone?: string;
  maxShots?: number;
}): string {
  const { season, our, opp, k = 5, wOff = 0.7, shotType, zone } = opts;

  const params = new URLSearchParams({
    season,
    our,
    opp,
    k: String(k),
    w_off: String(wOff),
  });

  if (shotType) params.set("shot_type", shotType);
  if (zone) params.set("zone", zone);

  return `${API_BASE}/export/shotplan.pdf?${params.toString()}`;
}

export async function fetchShotMlAnalysis(opts?: {
  nSplits?: number;
  refresh?: boolean;
}): Promise<ShotMlAnalysisResponse> {
  const nSplits = opts?.nSplits ?? 5;
  const params = new URLSearchParams();
  params.set("n_splits", String(nSplits));
  if (opts?.refresh) params.set("refresh", "true");

  const candidates = [
    `${API_BASE}/pbp/analysis/shot-ml?${params.toString()}`,
    `${API_BASE}/analysis/shot-ml?${params.toString()}`,
  ];

  const { data } = await tryJsonCandidates<any>(candidates, {
    keepTryingOnClientError: true,
  });

  return normalizeAnalysisResponse<ShotMlAnalysisResponse>(data);
}

export async function fetchShotModelMetrics(
  nSplits = 5,
  refresh = false
): Promise<ShotModelMetricsResponse> {
  const params = new URLSearchParams({ n_splits: String(nSplits) });
  if (refresh) params.set("refresh", "true");

  const candidates = [
    `${API_BASE}/pbp/metrics/shot-models?${params.toString()}`,
    `${API_BASE}/metrics/shot-models?${params.toString()}`,
  ];

  const { data } = await tryJsonCandidates<any>(candidates, {
    keepTryingOnClientError: true,
  });

  const base = unwrapMaybeCachedPayload(data);

  return {
    n_splits: Number((base as any)?.n_splits ?? nSplits),
    metrics: Array.isArray((base as any)?.metrics) ? (base as any).metrics : [],
  };
}

// ---------------------------
// Dataset2 Shots Explorer helpers
// ---------------------------

function buildPbpShotsParams(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}) {
  const params = new URLSearchParams();

  params.set("season", opts.season);
  params.set("limit", String(opts.limit ?? 50));
  params.set("team", opts.team);
  params.set("our", opts.team);

  if (opts.opp) params.set("opp", opts.opp);

  if (opts.shotType) {
    params.set("shot_type", opts.shotType);
    params.set("shotType", opts.shotType);
  }

  if (opts.zone) params.set("zone", opts.zone);

  return params;
}

export async function fetchPbpShotsPreview(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): Promise<PbpShotsPreviewResponse> {
  const params = buildPbpShotsParams(opts);

  const candidates = [
    `${API_BASE}/pbp/shots/preview?${params.toString()}`,
    `${API_BASE}/shots/preview?${params.toString()}`,
  ];

  const { data, usedUrl } = await tryJsonCandidates<PbpShotsPreviewResponse>(candidates, {
    keepTryingOnClientError: true,
  });

  const safe: PbpShotsPreviewResponse = {
    season: (data as any)?.season ?? opts.season,
    team: (data as any)?.team ?? (data as any)?.our ?? opts.team,
    opp: (data as any)?.opp ?? null,
    shot_type: (data as any)?.shot_type ?? null,
    zone: (data as any)?.zone ?? null,
    total_rows: Number((data as any)?.total_rows ?? 0),
    returned_rows: Number((data as any)?.returned_rows ?? (data as any)?.rows?.length ?? 0),
    columns: Array.isArray((data as any)?.columns) ? (data as any).columns : [],
    rows: Array.isArray((data as any)?.rows) ? (data as any).rows : [],
    _endpoint_used: usedUrl,
  };

  return safe;
}

export function getPbpShotsCsvUrl(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): string {
  const params = buildPbpShotsParams({
    ...opts,
    limit: opts.limit ?? 5000,
  });

  return `${API_BASE}/pbp/shots.csv?${params.toString()}`;
}

export function getPbpShotsCsvUrlLegacy(opts: {
  season: string;
  team: string;
  opp?: string;
  shotType?: string;
  zone?: string;
  limit?: number;
}): string {
  const params = buildPbpShotsParams({
    ...opts,
    limit: opts.limit ?? 5000,
  });

  return `${API_BASE}/shots.csv?${params.toString()}`;
}
