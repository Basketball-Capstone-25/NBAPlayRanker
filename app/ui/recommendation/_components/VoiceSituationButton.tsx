"use client";

// UI Layer — speak the game situation instead of using the controls (SCRUM-626).
// The browser's speech recognition gives the words; /nlp/parse finds the score
// margin, period and time left. Nothing runs until the coach checks the values.

import React, { useEffect, useRef, useState } from "react";
import {
  describeVoiceResult,
  situationFromParse,
  speechRecognitionFor,
  type ParseReply,
  type VoiceSituation,
} from "../../../domain/voice-situation";
import { parseSpokenSituation } from "../../../infrastructure/nlp-parse";

type Recognition = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type Status = "idle" | "listening" | "parsing";

/** Microphone icon so the button reads as voice input at a glance. */
function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </svg>
  );
}

export function voiceErrorMessage(error?: string): string {
  if (error === "not-allowed" || error === "service-not-allowed") {
    return "Microphone access is blocked. Allow it in the browser, or set the situation by hand.";
  }
  if (error === "no-speech") return "No speech was heard. Try again, closer to the microphone.";
  return "Voice input stopped. Try again, or set the situation by hand.";
}

export default function VoiceSituationButton({
  onResult,
  parse = parseSpokenSituation,
  supportedOverride,
}: {
  onResult: (result: VoiceSituation, heard: string) => void;
  parse?: (text: string) => Promise<ParseReply>;
  /** Tests only: render as if the browser does (or does not) support speech. */
  supportedOverride?: boolean;
}) {
  const [supported, setSupported] = useState<boolean | null>(supportedOverride ?? null);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const recognitionRef = useRef<Recognition | null>(null);

  useEffect(() => {
    if (supportedOverride === undefined) setSupported(speechRecognitionFor(window as never) !== null);
  }, [supportedOverride]);

  async function understand(heard: string) {
    setStatus("parsing");
    try {
      const result = situationFromParse(await parse(heard));
      onResult(result, heard);
      setMessage(describeVoiceResult(heard, result));
    } catch {
      setMessage(`Heard "${heard.trim()}", but it could not be understood. Set the situation by hand or try again.`);
    } finally {
      setStatus("idle");
    }
  }

  function start() {
    const Ctor = speechRecognitionFor(window as never);
    if (!Ctor) {
      setSupported(false);
      return;
    }
    const recognition = new Ctor() as Recognition;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const heard = event.results[0]?.[0]?.transcript ?? "";
      if (heard.trim()) void understand(heard);
    };
    recognition.onerror = (event) => {
      setMessage(voiceErrorMessage(event.error));
      setStatus("idle");
    };
    recognition.onend = () => setStatus((s) => (s === "listening" ? "idle" : s));
    recognitionRef.current = recognition;
    setMessage("");
    setStatus("listening");
    recognition.start();
  }

  function stop() {
    recognitionRef.current?.stop();
  }

  if (supported === false) {
    return (
      <p className="muted" style={{ fontSize: 12, margin: 0 }} data-testid="voice-unsupported">
        Voice input works in Chrome, Edge and Safari. Set the situation with the controls below.
      </p>
    );
  }

  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
      <button
        className={status === "listening" ? "btn btnPrimary" : "btn"}
        type="button"
        aria-pressed={status === "listening"}
        disabled={supported === null || status === "parsing"}
        onClick={status === "listening" ? stop : start}
        style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
      >
        <MicIcon />
        {status === "listening" ? "Listening… tap to stop" : status === "parsing" ? "Understanding…" : "Speak the situation"}
      </button>
      <span className="muted" style={{ fontSize: 12 }}>
        For example: “down 3, 40 seconds left, fourth quarter”
      </span>
      {message ? (
        <p role="status" style={{ flexBasis: "100%", margin: 0, fontSize: 13 }}>
          {message}
        </p>
      ) : null}
    </div>
  );
}
