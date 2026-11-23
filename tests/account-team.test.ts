import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { accountTeamFor, teamFromMetadata } from "../app/domain/account-team";

describe("account team (SCRUM-842)", () => {
  it("reads the saved team from user metadata", () => {
    expect(teamFromMetadata({ team: " tor " })).toBe("TOR");
    expect(teamFromMetadata({ requested_role: "coach" })).toBe("");
    expect(teamFromMetadata(null)).toBe("");
  });

  it("pre-fills only a team that exists in the loaded data", () => {
    expect(accountTeamFor("TOR", ["BOS", "TOR"])).toBe("TOR");
    expect(accountTeamFor("XYZ", ["BOS", "TOR"])).toBe("");
    expect(accountTeamFor("", ["BOS", "TOR"])).toBe("");
  });
});

describe("pages pre-fill only the account team (SCRUM-842)", () => {
  const pages = [
    "app/ui/recommendation/_components/MatchupPageClient.tsx",
    "app/ui/recommendation/_components/ContextPageClient.tsx",
    "app/ui/recommendation/_components/MatchupDashboardClient.tsx",
    "app/ui/gameplan/_components/GameplanClient.tsx",
    "app/ui/shot-analysis/_components/ShotPlanPageClient.tsx",
    "app/ui/shot-analysis/_components/ShotHeatmapPageClient.tsx",
    "app/ui/shot-analysis/_components/ShotExplorerPageClient.tsx",
  ];

  it.each(pages)("%s fills the team only when it is still blank", (file) => {
    const src = readFileSync(file, "utf8");
    expect(src).toContain("useAccountTeam(");
    expect(src).toMatch(/\(prev\) => prev \|\| accountTeam/);
  });

  it("keeps the team out of the authorization profile", () => {
    const auth = readFileSync("app/infrastructure/auth.ts", "utf8");
    expect(auth).toContain("supabase.auth.updateUser({ data: { team: team || null } })");
    expect(auth).not.toMatch(/from\("profiles"\)\s*\.update/);
  });
});
