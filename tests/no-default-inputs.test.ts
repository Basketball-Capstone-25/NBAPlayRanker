import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const read = (file: string) => readFileSync(`app/ui/recommendation/_components/${file}`, "utf8");

describe("Baseline and Context ML start with no game inputs (SCRUM-824)", () => {
  it.each(["MatchupPageClient.tsx", "ContextPageClient.tsx"])("%s pre-selects no matchup and does not run on load", (file) => {
    const src = read(file);
    expect(src).not.toMatch(/pickDefault(Season|Our|Opp)/);
    expect(src).not.toContain("didAutoRunRef");
    for (const prompt of ["Choose a season", "Choose our team", "Choose an opponent"]) {
      expect(src).toContain(`<option value="" disabled>${prompt}</option>`);
    }
    expect(src).toContain("(model setting)");
  });

  it("leaves the Context ML situation blank until the coach sets it", () => {
    const src = read("ContextPageClient.tsx");
    expect(src).toContain("useState<number | null>(null);");
    expect(src).not.toMatch(/useState<number>\((-?2|4|120)\)/);
    expect(src).toContain("return situationSet;");
    expect(src).toContain('useState("")');
    expect(src).toContain('<option value="" disabled>Period</option>');
  });

  it("still fills in values carried by a link", () => {
    expect(read("ContextPageClient.tsx")).toContain('linkedValue(qsSeason, m.seasons ?? [])');
    expect(read("MatchupPageClient.tsx")).toContain("linkedMatchup(window.location.search");
  });
});
