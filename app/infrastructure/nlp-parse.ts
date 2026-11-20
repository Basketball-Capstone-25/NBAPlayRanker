import { API_BASE, postJson } from "./api-client";
import type { ParseReply } from "../domain/voice-situation";

/** Send a spoken (or typed) game situation to /nlp/parse with no defaults (SCRUM-626). */
export function parseSpokenSituation(text: string): Promise<ParseReply> {
  return postJson<ParseReply>(`${API_BASE}/nlp/parse`, { text });
}
