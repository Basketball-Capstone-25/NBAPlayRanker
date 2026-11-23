import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { describeVoiceResult, situationFromParse, speechRecognitionFor } from "../app/domain/voice-situation";
import VoiceSituationButton, { voiceErrorMessage } from "../app/ui/recommendation/_components/VoiceSituationButton";

describe("spoken game situation (SCRUM-626)", () => {
  it("fills every field the phrase gave", () => {
    const result = situationFromParse({ context: { margin: -3, period: 4, time_remaining: 40 }, defaulted_fields: [], rejected_fields: [] });
    expect(result).toEqual({ margin: -3, period: 4, timeRemaining: 40, unclear: [] });
  });

  it("never fills a field the parser defaulted or rejected", () => {
    const result = situationFromParse({
      context: { margin: 0, period: 4, time_remaining: 720 },
      defaulted_fields: ["margin", "time_remaining"],
      rejected_fields: [],
    });
    expect(result).toEqual({ margin: null, period: 4, timeRemaining: null, unclear: ["margin", "timeRemaining"] });
    expect(situationFromParse({ context: { margin: 2, period: 9, time_remaining: 30 }, rejected_fields: ["period"] }).unclear).toEqual(["period"]);
  });

  it("treats missing or impossible values as unclear", () => {
    expect(situationFromParse({ context: { margin: null, period: "4", time_remaining: 900 } })).toEqual({
      margin: null, period: 4, timeRemaining: null, unclear: ["margin", "timeRemaining"],
    });
    expect(situationFromParse({}).unclear).toEqual(["margin", "period", "timeRemaining"]);
  });

  it("tells the coach what was heard and what is left to set", () => {
    const all = { margin: -3, period: 4, timeRemaining: 40, unclear: [] as never[] };
    expect(describeVoiceResult("down 3 40 seconds left", all)).toBe('Heard "down 3 40 seconds left". Check the filled values, then run.');
    expect(describeVoiceResult("down 3", { margin: -3, period: null, timeRemaining: null, unclear: ["period", "timeRemaining"] }))
      .toBe('Heard "down 3". Set the period and time remaining by hand, then run.');
    expect(describeVoiceResult("hello", { margin: null, period: null, timeRemaining: null, unclear: ["margin", "period", "timeRemaining"] }))
      .toContain("No score, period or time was understood");
  });

  it("finds the browser's speech recognition, or none (for example Firefox)", () => {
    class Speech {}
    expect(speechRecognitionFor({ webkitSpeechRecognition: Speech })).toBe(Speech);
    expect(speechRecognitionFor({ SpeechRecognition: Speech })).toBe(Speech);
    expect(speechRecognitionFor({})).toBeNull();
    expect(speechRecognitionFor(undefined)).toBeNull();
  });

  it("offers the button where speech works and the controls where it does not", () => {
    const withVoice = renderToStaticMarkup(<VoiceSituationButton onResult={() => {}} supportedOverride />);
    expect(withVoice).toContain("Speak the situation");
    // The microphone icon is decorative; the label still names the action.
    expect(withVoice).toMatch(/<svg[^>]*aria-hidden="true"/);
    const without = renderToStaticMarkup(<VoiceSituationButton onResult={() => {}} supportedOverride={false} />);
    expect(without).toContain("Voice input works in Chrome, Edge and Safari");
    expect(without).not.toContain("<button");
  });

  it("explains a blocked microphone", () => {
    expect(voiceErrorMessage("not-allowed")).toContain("Microphone access is blocked");
    expect(voiceErrorMessage("no-speech")).toContain("No speech was heard");
  });

  it("fills Context ML without running it and highlights what was not understood", () => {
    const src = readFileSync("app/ui/recommendation/_components/ContextPageClient.tsx", "utf8");
    expect(src).toContain("<VoiceSituationButton onResult={applyVoice} />");
    expect(src).toContain("setUnclear(result.unclear);");
    const applyVoice = src.slice(src.indexOf("function applyVoice"), src.indexOf("// Quick preset buttons."));
    expect(applyVoice).toContain("setMarginIn(result.margin);");
    expect(applyVoice).not.toContain("runContext");
    expect(src).toContain('needsHand("margin", marginIn)');
  });
});
