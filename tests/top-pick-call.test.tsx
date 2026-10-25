import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import TopPickCall from "../app/ui/_shared/components/TopPickCall";

describe("top-pick action label", () => {
  it("uses the first displayed ranking, not another play with a larger unrelated score", () => {
    const rows = [{ playType: "Spotup", score: 1.1 }, { playType: "Postup", score: 1.3 }];
    const html = renderToStaticMarkup(<TopPickCall rows={rows} />);
    expect(html).toContain("Run Spot-Up");
    expect(html).not.toContain("Post-Up");
    expect(rows[0].playType).toBe("Spotup");
  });
  it("changes with the ranked order", () => {
    const rows = [{ playType: "Spotup" }, { playType: "PRBallHandler" }];
    expect(renderToStaticMarkup(<TopPickCall rows={[...rows].reverse()} />)).toContain("Run Pick-and-Roll (Ball Handler)");
  });
  it("does not show a call for empty, loading or stale results", () => {
    expect(renderToStaticMarkup(<TopPickCall rows={[]} />)).toBe("");
    expect(renderToStaticMarkup(<TopPickCall rows={[{ playType: "Cut" }]} current={false} />)).toBe("");
  });
  it("keeps unfamiliar names as text instead of substituting another play", () => {
    expect(renderToStaticMarkup(<TopPickCall rows={[{ playType: "Custom <Cut>" }]} />)).toContain("Run Custom &lt;Cut&gt;");
  });
});
