import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));
import Gameplan, { GameplanShotAdvice, GameplanDiagramCaption } from "../app/ui/gameplan/_components/GameplanClient";

describe("Gameplan before a matchup is built", () => {
  it("does not export shot advice before a completed gameplan", () => {
    const html = renderToStaticMarkup(<Gameplan />);
    const pdfButton = html.match(/<button[^>]*>Export PDF<\/button>/)?.[0];
    expect(pdfButton).toBeDefined();
    expect(pdfButton).toContain('disabled=""');
  });

  it("shows a prompt instead of unearned shot advice", () => {
    const html = renderToStaticMarkup(<Gameplan />);
    expect(html).toContain("Build a gameplan to see shot priorities for this matchup.");
    expect(html).not.toContain("We want a high-quality shot");
    expect(html).not.toContain("Primary: get a clean rim touch");
    expect(html).not.toContain("Dataset2 endpoint");
  });
});

describe("Gameplan evidence scope", () => {
  it("describes a season-ranked dunk without claiming it satisfies a need for three points", () => {
    const html = renderToStaticMarkup(<GameplanShotAdvice shotType="Dunk" zone="rim" />);
    expect(html).toContain("Selected shot type: Dunk.");
    expect(html).toContain("Selected court zone: rim.");
    expect(html).toContain("season-long offense and opponent defense");
    expect(html).toContain("do not use the score, clock or coaching conditions");
    expect(html).toContain("points per shot attempt, not points per possession");
    expect(html).not.toContain("open 3");
    expect(html).not.toContain("Vs switch");
  });
  it("distinguishes the play diagram baseline from the Smart ranking", () => {
    const html = renderToStaticMarkup(<GameplanDiagramCaption caption="PRRollMan: Pred 1.079 PPP." />);
    expect(html).toContain("Pred 1.079 PPP");
    expect(html).toContain("Season-baseline diagram");
    expect(html).toContain("does not include the game situation or coaching adjustments");
  });
});
