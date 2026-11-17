import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const read = (file: string) => readFileSync(`app/ui/data-explorer/${file}`, "utf8");

describe("Data Explorer export buttons say what they export (SCRUM-393)", () => {
  it("names the play-type rows, matching shots and clutch report", () => {
    expect(read("DataExplorerClient.tsx").match(/"Export play-type rows \(CSV\)"/g)).toHaveLength(2);
    const shots = read("ShotDataExplorer.tsx");
    expect(shots).toContain('"Export matching shots (CSV)"');
    expect(shots).toContain('"Export matching shots (JSON)"');
    const clutch = read("ClutchReportPanel.tsx");
    expect(clutch).toContain('"Download clutch report (PDF)"');
    expect(clutch).toContain('"Download clutch report (CSV)"');
  });

  it("leaves no export button with only a file format", () => {
    for (const file of ["DataExplorerClient.tsx", "ShotDataExplorer.tsx", "ClutchReportPanel.tsx"]) {
      expect(read(file)).not.toMatch(/"(Export|Download) (CSV|JSON|PDF)"/);
    }
  });
});
