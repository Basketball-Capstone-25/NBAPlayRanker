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
    const data = report(); data.team_possessions = data.opponent_possessions = 1_000_000;
    data.rows = [{ play_type: "Cut", team: { possessions: 1_000_000, points: 999_999, usage_pct: 100, ppp: .999999 },
      opponent: { possessions: 1_000_000, points: 1_000_000, usage_pct: 100, ppp: 1 }, difference: { usage_pp: 0, ppp: -.000001 } }];
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

describe("team profile readouts", () => {
  const mainView = (data: TeamComparison, metric: "ppp" | "usage", minimum = 20) =>
    renderToStaticMarkup(<TeamComparisonTable data={data} minimum={minimum} initialMetric={metric} />).split("<details")[0];

  it.each([false, true])("rounds the source-count difference consistently when teams are reversed: %s", reverse => {
    const data = report(); data.team_possessions = 80; data.opponent_possessions = 100;
    data.rows = [{ play_type: "OffScreen", team: { possessions: 80, points: 91, usage_pct: 100, ppp: 91 / 80 },
      opponent: { possessions: 100, points: 100, usage_pct: 100, ppp: 1 }, difference: { usage_pp: 0, ppp: 91 / 80 - 1 } }];
    if (reverse) {
      [data.rows[0].team, data.rows[0].opponent] = [data.rows[0].opponent, data.rows[0].team];
      [data.team_possessions, data.opponent_possessions] = [data.opponent_possessions, data.team_possessions];
      data.rows[0].difference.ppp *= -1;
    }
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} minimum={1} />);
    expect(html).toContain("13.8 points apart");
    expect(html).toContain(`${reverse ? "BOS" : "TOR"} scored 13.8 more points`);
    expect(html).toContain(reverse ? "-0.138" : "+0.138");
    expect(html).not.toContain("13.7 points apart");
  });

  it("converts PPP to points per 100 play-type possessions with the same bar scale", () => {
    const html = mainView(report(), "ppp", 1);
    expect(html.match(/<tr[ >]/g)).toHaveLength(3);
    expect(html).toContain('data-label="TOR"'); expect(html).toContain('data-label="BOS"');
    expect(html).toContain("120.0"); expect(html).toContain("100.0");
    expect(html).toContain("20.0 points apart");
    expect(html).toContain("TOR scored 20.0 more points per 100 of these possessions");
    expect(html).toContain('style="width:100%"');
    expect(html).toContain('style="width:83.33333333333333%"');
    expect(html).not.toContain("1.200");
    expect(html).toContain("does not mean running more of that play will raise team scoring");
  });

  it("uses the minimum for the headline without hiding a smaller sample", () => {
    const html = mainView(report(), "ppp");
    expect(html).toContain("Isolation: BOS scored 10.0 more points per 100 of these possessions");
    expect(html).toContain("Small sample"); expect(html).toContain("120.0");
    expect(html).not.toContain("Cut: TOR scored 20.0");
    expect(html).toContain("Too few possessions to highlight a difference");
  });

  it("presents usage as a style comparison with percentage-point differences", () => {
    const html = mainView(report(), "usage", 1);
    expect(html).toContain("10.0%"); expect(html).toContain("20.0%");
    expect(html).toContain("10.0 pp apart"); expect(html).toContain("BOS used this play more often");
    expect(html).toContain("style, not which team is better");
    expect(html).not.toContain('data-tone="favorable"');
    expect(html).not.toContain('data-tone="caution"');
    expect(html).toContain('style="width:10%"'); expect(html).toContain('style="width:20%"');
  });

  it("keeps missing scoring rates and real zero usage separate", () => {
    const data = report();
    data.rows = [{ play_type: "Cut", team: { possessions: 0, points: 0, usage_pct: 0, ppp: null },
      opponent: null, difference: { usage_pp: null, ppp: null } }];
    const scoring = mainView(data, "ppp");
    expect(scoring).toContain("Comparison unavailable"); expect(scoring).not.toContain("<strong>0.0</strong>");
    expect(scoring).toContain("More shared data is needed");
    expect(scoring).toContain("<small>Unavailable</small>");
    expect(scoring).not.toContain('aria-hidden="true"');
    const usage = mainView(data, "usage");
    expect(usage).toContain("<strong>0.0%</strong>"); expect(usage).toContain("Comparison unavailable");
    expect(usage.match(/aria-hidden="true"/g)).toHaveLength(1);
    expect(usage).toContain('style="width:0%"');
  });

  it("swaps the interpretation when both teams and values are swapped", () => {
    const data = report(); data.filters.team = "BOS"; data.filters.opponent = "TOR";
    [data.team_possessions, data.opponent_possessions] = [data.opponent_possessions, data.team_possessions];
    for (const row of data.rows) {
      [row.team, row.opponent] = [row.opponent, row.team];
      row.difference = { usage_pp: -row.difference.usage_pp!, ppp: -row.difference.ppp! };
    }
    const html = mainView(data, "ppp", 1);
    expect(html).toContain("BOS and TOR: how they play");
    expect(html).toContain("Cut: TOR scored 20.0 more points per 100 of these possessions");
  });

  it("uses new season data and retains exact points in the full numbers", () => {
    const data = report(); data.filters.season = "2026-27";
    data.rows[0].team!.points = 15; data.rows[0].team!.ppp = 1.5; data.rows[0].difference.ppp = 0.5;
    const html = renderToStaticMarkup(<TeamComparisonTable data={data} minimum={1} />);
    expect(html).toContain("2026-27"); expect(html).not.toContain("2024-25");
    expect(html).toContain("150.0"); expect(html).toContain("50.0 points apart");
    expect(html).toContain("<td>1.500</td><td>15</td>");
    expect(html).toContain("not a count of every team possession");
    expect(html).toContain("not whole-team offensive rating");
  });
});
