// app/domain/voice-situation.ts
//
// Domain Layer — turning a spoken game situation into Context ML inputs
// (SCRUM-626). The browser turns speech into text; /nlp/parse turns the text
// into fields. These rules decide which fields were really understood, so a
// field the coach did not say is never filled with a guess.

export type SituationField = "margin" | "period" | "timeRemaining";

export type ParseReply = {
  context?: Record<string, unknown>;
  defaulted_fields?: string[];
  rejected_fields?: string[];
};

export type VoiceSituation = {
  margin: number | null;
  period: number | null;
  timeRemaining: number | null;
  /** Fields the phrase did not give clearly; the page highlights them. */
  unclear: SituationField[];
};

const API_FIELD: Record<SituationField, string> = {
  margin: "margin",
  period: "period",
  timeRemaining: "time_remaining",
};

const LIMITS: Record<SituationField, [number, number]> = {
  margin: [-40, 40],
  period: [1, 5],
  timeRemaining: [0, 720],
};

export const FIELD_LABEL: Record<SituationField, string> = {
  margin: "score margin",
  period: "period",
  timeRemaining: "time remaining",
};

/** Keep only the fields the parser found itself and that fit on a scoreboard. */
export function situationFromParse(reply: ParseReply): VoiceSituation {
  const skipped = new Set([...(reply.defaulted_fields ?? []), ...(reply.rejected_fields ?? [])]);
  const out: VoiceSituation = { margin: null, period: null, timeRemaining: null, unclear: [] };
  for (const field of ["margin", "period", "timeRemaining"] as SituationField[]) {
    const name = API_FIELD[field];
    const raw = reply.context?.[name];
    const value = typeof raw === "number" ? raw : typeof raw === "string" && raw.trim() !== "" ? Number(raw) : NaN;
    const [lo, hi] = LIMITS[field];
    if (skipped.has(name) || !Number.isFinite(value) || value < lo || value > hi) {
      out.unclear.push(field);
      continue;
    }
    out[field] = Math.round(value);
  }
  return out;
}

/** What the page tells the coach after a phrase is understood. */
export function describeVoiceResult(heard: string, result: VoiceSituation): string {
  const quoted = `Heard "${heard.trim()}".`;
  if (result.unclear.length === 0) return `${quoted} Check the filled values, then run.`;
  if (result.unclear.length === 3) return `${quoted} No score, period or time was understood. Set them below or try again.`;
  const list = result.unclear.map((f) => FIELD_LABEL[f]);
  const names = list.length === 1 ? list[0] : `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`;
  return `${quoted} Set the ${names} by hand, then run.`;
}

type SpeechWindow = { SpeechRecognition?: unknown; webkitSpeechRecognition?: unknown };

/** The browser's speech recognition, or null where it is not supported (for example Firefox). */
export function speechRecognitionFor(win: SpeechWindow | undefined): (new () => unknown) | null {
  const ctor = win?.SpeechRecognition ?? win?.webkitSpeechRecognition;
  return typeof ctor === "function" ? (ctor as new () => unknown) : null;
}
