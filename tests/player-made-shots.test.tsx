import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { cellOpacity, describeMadeShots, heatCells, listSeasons, toSvgX, toSvgY } from "../app/domain/player-shots";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));
import PlayerMadeShotsPanel, { CourtHeatmap } from "../app/ui/recommendation/_components/PlayerMadeShotsPanel";

const result = {
  player: { id: "10", name: "Scottie Barnes" }, our: "TOR", opp: "BOS", seasons: ["2023-24", "2024-25"],
  games: 3, player_games: 3, attempts: 9, made: 4, shown: 3, excluded_location: 1, outside_halfcourt: 0,
  points: [[41.0, 0.0], [40.5, 0.5], [20.0, 10.0]] as [number, number][],
};

describe("player made-shot heat map (SCRUM-825)", () => {
  it("counts made shots in 2 ft squares and shades the busiest deepest", () => {
    const cells = heatCells(result.points);
    expect(cells).toHaveLength(2);
    expect(cells.find((c) => c.count === 2)?.share).toBe(1);
    expect(cells.find((c) => c.count === 1)?.share).toBe(0.5);
    expect(cellOpacity(1)).toBeCloseTo(0.95);
    expect(cellOpacity(0)).toBeCloseTo(0.2);
    expect(heatCells([[60, 0], [10, 40], [Number.NaN, 0]])).toEqual([]);
  });

  it("puts the basket at the top of the court", () => {
    expect(toSvgY(41.75)).toBeCloseTo(52.5);
    expect(toSvgY(0)).toBe(470);
    expect(toSvgX(0)).toBe(250);
  });

  it("states the made shots, games and what is hidden", () => {
    expect(listSeasons(["2022-23", "2023-24", "2024-25"])).toBe("2022-23, 2023-24 and 2024-25");
    expect(describeMadeShots(result)).toBe(
      "Scottie Barnes: 4 made shots in 3 games against BOS in 2023-24 and 2024-25. 1 without a usable court location is not shown.",
    );
    expect(describeMadeShots({ ...result, made: 0, shown: 0 })).toBe("Scottie Barnes has no made shots against BOS in 2023-24 and 2024-25.");
  });

  it("draws green squares on a court", () => {
    const html = renderToStaticMarkup(<CourtHeatmap result={result} />);
    expect(html.match(/fill="#16a34a"/g)).toHaveLength(2);
    expect(html).toContain('role="img"');
  });

  it("waits for a matchup and pre-selects no season or player", () => {
    expect(renderToStaticMarkup(<PlayerMadeShotsPanel our="" opp="" />)).toContain("Choose our team and the opponent above.");
    const html = renderToStaticMarkup(<PlayerMadeShotsPanel our="TOR" opp="BOS" />);
    expect(html).toContain("Choose seasons first");
    expect(html).not.toContain('checked=""');
  });

  it("is shown on Context ML for the chosen matchup", () => {
    expect(readFileSync("app/ui/recommendation/_components/ContextPageClient.tsx", "utf8")).toContain("<PlayerMadeShotsPanel our={our} opp={opp} />");
  });
});
