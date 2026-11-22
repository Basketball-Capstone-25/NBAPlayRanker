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

describe("every other page starts with no game inputs (SCRUM-841)", () => {
  const pages = [
    "app/ui/gameplan/_components/GameplanClient.tsx",
    "app/ui/recommendation/_components/MatchupDashboardClient.tsx",
    "app/ui/data-explorer/DataExplorerClient.tsx",
    "app/ui/data-explorer/ShotDataExplorer.tsx",
    "app/ui/data-explorer/AnalystSeasonSummary.tsx",
    "app/ui/shot-analysis/_components/ShotPlanPageClient.tsx",
    "app/ui/shot-analysis/_components/ShotHeatmapPageClient.tsx",
    "app/ui/shot-analysis/_components/ShotExplorerPageClient.tsx",
    "app/ui/analytics/_components/TopKUpliftPanel.tsx",
  ];

  it.each(pages)("%s pre-selects no team, season or situation", (file) => {
    const src = readFileSync(file, "utf8");
    expect(src).not.toMatch(/pickDefault(Season|Team|Our|Opp)/);
    expect(src).not.toMatch(/useState(<string>)?\("(TOR|BOS|20\d\d-\d\d)"\)/);
    expect(src).not.toMatch(/includes\("(TOR|BOS)"\)/);
    expect(src).not.toMatch(/seasons\[(meta\.|result\.|m\.)?seasons\.length - 1\]/);
  });

  it("starts the gameplan situation blank", () => {
    const src = readFileSync(pages[0], "utf8");
    expect(src).toContain("useState<number | null>(null)");
    expect(src).not.toMatch(/useState\((4|180|-2)\)/);
    expect(src).toContain('useState<string>("")');
  });
});
