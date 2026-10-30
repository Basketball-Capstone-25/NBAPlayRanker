import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { topPickIndexes } from "../app/domain/top-pick";
import TopPickBadge from "../app/ui/_shared/components/TopPickBadge";

describe("#1 play highlight (SCRUM-496)", () => {
  it("marks only the first ranked play when it is clearly ahead", () => {
    expect(topPickIndexes([1.21, 1.18, 1.05])).toEqual([0]);
  });

  it("marks every play tied with #1 on the displayed score", () => {
    expect(topPickIndexes([1.1234, 1.1231, 1.0, 1.1229])).toEqual([0, 1, 3]);
    expect(topPickIndexes(["1.2", 1.2, 1.19])).toEqual([0, 1]);
  });

  it("still marks the first row when scores are missing, and nothing for empty results", () => {
    expect(topPickIndexes([null, null])).toEqual([0]);
    expect(topPickIndexes([])).toEqual([]);
  });

  it("labels a single top pick and a tie differently", () => {
    expect(renderToStaticMarkup(<TopPickBadge tied={false} />)).toContain("Top pick");
    expect(renderToStaticMarkup(<TopPickBadge tied />)).toContain("Tied #1");
  });

  it("is used on both the Matchup and Context rankings", () => {
    for (const file of ["MatchupPageClient.tsx", "ContextPageClient.tsx"]) {
      const src = readFileSync(`app/ui/recommendation/_components/${file}`, "utf8");
      expect(src).toContain("topPickIndexes(");
      expect(src).toContain('"barRow topPickRow"');
      expect(src).toContain("<TopPickBadge tied={topTied} />");
    }
  });

  it("keeps the highlight readable in light, dark and high-contrast modes", () => {
    const css = readFileSync("app/globals.css", "utf8");
    expect(css).toMatch(/\.topPickBadge\s*{[^}]*background:\s*#fbbf24;[^}]*color:\s*#1f1300;/);
    expect(css).toContain("@media (prefers-color-scheme: dark)");
    expect(css).toContain("@media (forced-colors: active)");
  });
});
