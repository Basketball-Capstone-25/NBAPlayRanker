import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { TeamComparison } from "../app/domain/team-comparison";
import { TeamComparisonTable } from "../app/ui/recommendation/_components/TeamComparisonTable";
import TeamComparisonPanel, { TeamComparisonResultState } from "../app/ui/recommendation/_components/TeamComparisonPanel";

vi.mock("../app/infrastructure/team-comparison", () => ({ fetchTeamComparison: vi.fn() }));
const report = (): TeamComparison => ({
  status: "ok", filters: { season: "2024-25", team: "TOR", opponent: "BOS", side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "regular-season", team_possessions: 100, opponent_possessions: 200,
  rows: [{ play_type: "Cut",
    team: { possessions: 10, points: 12, usage_pct: 10, ppp: 1.2 },
    opponent: { possessions: 40, points: 40, usage_pct: 20, ppp: 1 },
    difference: { usage_pp: -10, ppp: 0.2 } },
  { play_type: "Isolation",
    team: { possessions: 90, points: 90, usage_pct: 90, ppp: 1 },
    opponent: { possessions: 160, points: 176, usage_pct: 80, ppp: 1.1 },
    difference: { usage_pp: 10, ppp: -0.1 } }],
});

describe("team profile table", () => {
  it("shows independent totals, signed differences and the season-long scope", () => {
    const html = renderToStaticMarkup(<TeamComparisonTable data={report()} />);
    for (const text of ["TOR and BOS season profiles", "TOR 100 · BOS 200", "10.0%", "20.0%", "1.200", "1.000",
      "-10.0 pp", "+10.0 pp", "+0.200", "-0.100", "Small sample", "not results from games against each other or a forecast",
      "Each team uses its own total", "PPP = points ÷ possessions", "fixture-v1"])
      expect(html).toContain(text);
    expect(html).toContain('rowSpan="3"');
    expect(html).toContain('scope="rowgroup"');
  });
  it("changes warnings without changing the calculated values", () => {
    const html = renderToStaticMarkup(<TeamComparisonTable data={report()} minimum={1} />);
    expect(html).not.toContain('>Small sample</span>');
    expect(html).toContain("10.0%"); expect(html).toContain("-10.0 pp");
  });
  it("keeps a missing category separate from a recorded zero", () => {
    const data = report();
    data.rows[0].team = { possessions: 0, points: 0, usage_pct: 0, ppp: null };
    data.rows[0].opponent = null;
    data.rows[0].difference = { usage_pp: null, ppp: null };
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} />);
    expect(html).toContain('<td>0</td><td>0.0%</td><td>—</td>');
    expect(html).toContain('scope="row">BOS</th><td>—</td><td>—</td><td>—</td>');
    expect(html).toContain('scope="row">Difference</th><td>—</td><td>—</td><td>—</td>');
  });
  it("retains the available team's rows when the other team has no data", () => {
    const data = report(); data.status = "no_data"; data.opponent_possessions = null;
    for (const row of data.rows) { row.opponent = null; row.difference = { usage_pp: null, ppp: null }; }
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} />);
    expect(html).toContain("No recorded offensive data for BOS in 2024-25");
    expect(html).toContain("TOR 100 · BOS —"); expect(html).toContain("Cut"); expect(html).toContain("1.200");
    expect(html).toContain("<table");
    data.team_possessions = null; data.rows = [];
    const empty = renderToStaticMarkup(<TeamComparisonTable data={data} />);
    expect(empty).toContain("No recorded offensive data for TOR or BOS"); expect(empty).not.toContain("<table");
  });
  it("swaps team labels, values and the sign of each difference", () => {
    const data = report(); data.filters.team = "BOS"; data.filters.opponent = "TOR";
    [data.team_possessions, data.opponent_possessions] = [data.opponent_possessions, data.team_possessions];
    for (const row of data.rows) {
      [row.team, row.opponent] = [row.opponent, row.team];
      row.difference.usage_pp = -row.difference.usage_pp!; row.difference.ppp = -row.difference.ppp!;
    }
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} />);
    expect(html).toContain("BOS 200 · TOR 100"); expect(html).toContain("Difference = BOS minus TOR");
    expect(html).toContain('scope="row">BOS</th><td>40</td><td>20.0%</td><td>1.000</td>');
    expect(html).toContain("-0.200"); expect(html).toContain("+0.100");
  });
  it("does not show negative zero when differences round to zero", () => {
    const data = report(); data.rows[0].difference = { usage_pp: -0.00001, ppp: -0.00001 };
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} />);
    expect(html).toContain("0.0 pp"); expect(html).toContain("0.000");
    expect(html).not.toContain("-0.0 pp"); expect(html).not.toContain("-0.000");
  });
  it("hides old results during loading or failure and offers retry", () => {
    const state = { data: report(), loading: true, error: "" };
    const loading = renderToStaticMarkup(<TeamComparisonResultState state={state} minimum={20} retry={() => {}} />);
    expect(loading).toContain("Loading team profiles"); expect(loading).not.toContain("<table");
    const failed = renderToStaticMarkup(<TeamComparisonResultState state={{ ...state, loading: false, error: "Service unavailable" }} minimum={20} retry={() => {}} />);
    expect(failed).toContain('role="alert"'); expect(failed).toContain("Service unavailable");
    expect(failed).toContain("Retry comparison"); expect(failed).not.toContain("<table");
  });
  it("asks for valid selections before loading", () => {
    const filters = { season: "2024-25", team: "TOR", opponent: "TOR" };
    expect(renderToStaticMarkup(<TeamComparisonPanel filters={filters} minimum={20} />)).toContain("Choose two different teams");
    expect(renderToStaticMarkup(<TeamComparisonPanel filters={{ ...filters, opponent: "" }} minimum={20} />))
      .toContain("Select two teams and a season");
    expect(renderToStaticMarkup(<TeamComparisonPanel filters={{ ...filters, opponent: "BOS" }} minimum={20} />))
      .toContain("Loading team profiles");
  });
});
