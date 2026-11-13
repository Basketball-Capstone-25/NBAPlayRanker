import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { SeasonTrend } from "../app/domain/season-trend";
import { SeasonTrendChart } from "../app/ui/recommendation/_components/SeasonTrendChart";
import { SeasonTrendTable } from "../app/ui/recommendation/_components/SeasonTrendTable";

const report = (): SeasonTrend => ({
  status: "ok", filters: { team: "TOR", play_type: "Cut", side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture", aggregation_version: "regular-season",
  rows: [
    { season: "2021-22", possessions: 100, points: 120, team_possessions: 400, usage_pct: 25, ppp: 1.2 },
    { season: "2022-23", possessions: 10, points: 11, team_possessions: 100, usage_pct: 10, ppp: 1.1 },
    { season: "2023-24", possessions: null, points: null, team_possessions: 500, usage_pct: null, ppp: null },
    { season: "2024-25", possessions: 40, points: 60, team_possessions: 200, usage_pct: 20, ppp: 1.5 },
  ],
});
const trendPath = (html: string) => html.match(/<path d="([^"]*)"/)![1];

describe("season trend chart and table", () => {
  it("breaks the line at missing seasons without moving later points left", () => {
    const html = renderToStaticMarkup(<SeasonTrendChart data={report()} metric="ppp" minimum={20} />);
    const path = trendPath(html);
    expect(path.match(/M /g)).toHaveLength(2); expect(path.match(/L /g)).toHaveLength(1);
    expect(path).toContain("M 60 "); expect(path).toContain("M 740 ");
    expect(path).toContain("M 60 86"); // 1.2 PPP on the 0–1.5 scale.
    expect(path).toContain("M 740 50"); // 1.5 PPP reaches the labeled upper tick.
    expect(html.match(/<circle /g)).toHaveLength(3);
    expect(html).toContain("No data"); expect(html).toContain("2023-24");
    expect(html).toContain('role="img"'); expect(html).toContain("Exact values are in the table");
  });
  it("scales usage from zero to the next five-percentage-point boundary", () => {
    const html = renderToStaticMarkup(<SeasonTrendChart data={report()} metric="usage_pct" minimum={20} />);
    expect(html).toContain('cx="60" cy="50"');
    expect(html).toContain("25%"); expect(html).toContain("25.0% usage; 100 possessions");
    expect(html).toContain("6.25%");
  });
  it("makes small usage changes readable while preserving their true coordinates", () => {
    const data = report();
    data.rows = [
      { season: "2023-24", possessions: 59, points: 71, team_possessions: 1000, usage_pct: 5.9, ppp: 71 / 59 },
      { season: "2024-25", possessions: 87, points: 104, team_possessions: 1000, usage_pct: 8.7, ppp: 104 / 87 },
    ];
    const html = renderToStaticMarkup(<SeasonTrendChart data={data} metric="usage_pct" minimum={20} />);
    const coordinates = [...html.matchAll(/<circle cx="([^"]+)" cy="([^"]+)"/g)].map(match => [Number(match[1]), Number(match[2])]);
    expect(html).toContain("10%"); expect(html).toContain("0%"); expect(html).toContain("2.5%");
    expect(coordinates[0][0]).toBe(60); expect(coordinates[0][1]).toBeCloseTo(123.8);
    expect(coordinates[1][0]).toBe(740); expect(coordinates[1][1]).toBeCloseTo(73.4);
  });
  it("uses hollow circles for small samples without changing the line", () => {
    const small = renderToStaticMarkup(<SeasonTrendChart data={report()} metric="ppp" minimum={20} />);
    const all = renderToStaticMarkup(<SeasonTrendChart data={report()} metric="ppp" minimum={1} />);
    expect(small).toContain('fill="white"'); expect(all).not.toContain('fill="white"');
    expect(trendPath(small)).toBe(trendPath(all));
  });
  it("keeps zero usage while leaving zero-possession PPP unavailable", () => {
    const data = report(); data.rows = [{ season: "2024-25", possessions: 0, points: 0,
      team_possessions: 100, usage_pct: 0, ppp: null }]; data.status = "no_data";
    const usage = renderToStaticMarkup(<SeasonTrendChart data={data} metric="usage_pct" minimum={20} />);
    const ppp = renderToStaticMarkup(<SeasonTrendChart data={data} metric="ppp" minimum={20} />);
    expect(usage).toContain('cx="400" cy="230"'); expect(ppp).not.toContain("<circle");
    expect(renderToStaticMarkup(<SeasonTrendTable data={data} minimum={20} />)).toContain("<td>0.0%</td><td>—</td>");
  });
  it("shows the same rounded values and counts in the accessible table", () => {
    const html = renderToStaticMarkup(<SeasonTrendTable data={report()} minimum={20} />);
    for (const text of ["TOR", "Cut", "100", "120", "400", "25.0%", "1.200", "Small sample", "Not recorded", "2023-24"])
      expect(html).toContain(text);
    expect(html).toContain('scope="col"'); expect(html).toContain('scope="row"');
    expect(html).toContain('<td>—</td><td>—</td><td>500</td><td>—</td><td>—</td>');
  });
});
