import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { contextReason } from "../app/domain/context-analysis";
import RankedPlayList from "../app/ui/recommendation/_components/RankedPlayList";

const plays = [
  { playType: "PRBallHandler", ppp: 1.084, shotZone: "Mid-range", reason: "+0.05 PPP vs season baseline" },
  { playType: "Spotup", ppp: 1.0612, shotZone: "Corner 3", reason: "+0.02 PPP vs season baseline" },
  { playType: "Cut", ppp: 1.04, reason: "-0.01 PPP vs season baseline" },
];

describe("Context ML ranked list (SCRUM-795)", () => {
  it("shows rank, a Run call on #1, play, shot zone, reason and expected PPP", () => {
    const html = renderToStaticMarkup(<RankedPlayList plays={plays} current topPicks={new Set([0])} />);
    expect(html).toContain('aria-label="Ranked plays"');
    expect(html.match(/<li/g)).toHaveLength(3);
    expect(html).toContain("Run Pick-and-Roll (Ball Handler)");
    expect(html).toContain("Shot zone: Mid-range");
    expect(html).toContain("Shot zone: Corner 3");
    expect(html).toContain("+0.05 PPP vs season baseline");
    expect(html).toContain("1.08");
    expect(html).toContain("1.06");
    expect(html.match(/expected PPP/g)).toHaveLength(3);
    expect(html.match(/data-testid="top-pick-badge"/g)).toHaveLength(1);
  });

  it("skips the shot zone line when a play has none", () => {
    const html = renderToStaticMarkup(<RankedPlayList plays={[plays[2]]} current topPicks={new Set([0])} />);
    expect(html).not.toContain("Shot zone:");
  });

  it("marks every tied top play", () => {
    const html = renderToStaticMarkup(<RankedPlayList plays={plays} current topPicks={new Set([0, 1])} tied />);
    expect(html.match(/Tied #1/g)).toHaveLength(2);
    expect(html.match(/topPickRow/g)).toHaveLength(2);
  });

  it("hides the Run call while the situation has changed", () => {
    const html = renderToStaticMarkup(<RankedPlayList plays={plays} current={false} topPicks={new Set([0])} />);
    expect(html).not.toContain("top-pick-action");
  });

  it("renders nothing for empty results", () => {
    expect(renderToStaticMarkup(<RankedPlayList plays={[]} current topPicks={new Set()} />)).toBe("");
  });
});

describe("one-line reason", () => {
  it("compares to the season baseline and names a late-game state", () => {
    expect(contextReason(0.0512, "Late & trailing | Q4 0:28")).toBe("+0.05 PPP vs season baseline (late & trailing)");
    expect(contextReason(-0.031, "Normal context | Q2")).toBe("-0.03 PPP vs season baseline");
    expect(contextReason(0, "")).toBe("+0.00 PPP vs season baseline");
  });

  it("falls back when the change is missing", () => {
    expect(contextReason(Number.NaN, "Late & trailing")).toBe("Highest expected points for this matchup");
  });
});
