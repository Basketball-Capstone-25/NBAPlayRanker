import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const read = (file: string) => readFileSync(file, "utf8");

describe("no results appear until the user runs a page (SCRUM-843)", () => {
  it("shows matchup dashboard results only after Show results", () => {
    const src = read("app/ui/recommendation/_components/MatchupDashboardClient.tsx");
    expect(src).toContain("onClick={() => setShown(choice)}>Show results</button>");
    expect(src).toContain("if (shownTable && shownFilters) void current.run(shownFilters");
    expect(src).toContain("Inputs changed. Click Show results to update these results.");
  });

  it("computes the Top-K uplift only for the inputs that were run", () => {
    const src = read("app/ui/analytics/_components/TopKUpliftPanel.tsx");
    expect(src).toContain("onClick={() => setRan({ season, our, opp, k })}>Compute uplift</button>");
    expect(src).toContain("}, [ran]);");
    expect(src).toContain("Inputs changed. Click Compute uplift to update these results.");
  });

  it("never restores auto-render on the shot heatmap", () => {
    const src = read("app/ui/shot-analysis/_components/ShotHeatmapPageClient.tsx");
    expect(src).not.toContain("setAutoRender(parsed.autoRender");
    expect(src).toContain("useState<boolean>(false)");
  });

  it("does not run Baseline or Context ML on page load", () => {
    for (const file of ["MatchupPageClient.tsx", "ContextPageClient.tsx"]) {
      expect(read(`app/ui/recommendation/_components/${file}`)).not.toContain("didAutoRunRef");
    }
    // Live updates on Context ML are opt-in.
    expect(read("app/ui/recommendation/_components/ContextPageClient.tsx")).toContain("const [liveMode, setLiveMode] = useState(false);");
  });
});
