// app/services/gameplan.ts
//
// Application Layer — Gameplan Service subsystem
// Data transformation/normalization of backend responses for the Gameplan UI.
// Extracted from GameplanClient.tsx.

export type TeamOption = { code: string; name?: string };

export type MetaOptions = {
  seasons?: string[];
  teams?: string[];
  teamNames?: Record<string, string>;
  [key: string]: any;
};

export type NormalizedPlay = {
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

function toNumMaybe(v: any): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function normalizeTeams(meta: MetaOptions | null): TeamOption[] {
  const teams = (meta?.teams ?? []) as any[];
  const teamNames = (meta?.teamNames ?? {}) as Record<string, string>;
  return teams
    .map((t) => String(t))
    .filter(Boolean)
    .map((code) => ({ code, name: teamNames[code] }));
}

export function normalizeWhy(raw: any): string[] {
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

export function normalizePlay(
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
    toNumMaybe(raw?.finalPPP) ??
    toNumMaybe(raw?.PPP_PRED) ??
    toNumMaybe(raw?.pppPred) ??
    toNumMaybe(raw?.PPP_ML_BLEND) ??
    toNumMaybe(raw?.mlPPP) ??
    toNumMaybe(raw?.ppp) ??
    toNumMaybe(raw?.pred_ppp) ??
    null;

  const edge =
    toNumMaybe(raw?.DELTA_VS_BASELINE) ??
    toNumMaybe(raw?.deltaPPP) ??
    toNumMaybe(raw?.PPP_GAP) ??
    toNumMaybe(raw?.pppGap) ??
    toNumMaybe(raw?.edge) ??
    null;

  const offense =
    toNumMaybe(raw?.PPP_ML_BLEND) ??
    toNumMaybe(raw?.mlPPP) ??
    toNumMaybe(raw?.PPP_OFF_SHRUNK) ??
    toNumMaybe(raw?.pppOff) ??
    toNumMaybe(raw?.offense) ??
    null;

  const defense =
    toNumMaybe(raw?.PPP_BASELINE) ??
    toNumMaybe(raw?.baselinePPP) ??
    toNumMaybe(raw?.PPP_DEF_SHRUNK) ??
    toNumMaybe(raw?.pppDef) ??
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
