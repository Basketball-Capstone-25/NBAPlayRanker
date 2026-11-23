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
