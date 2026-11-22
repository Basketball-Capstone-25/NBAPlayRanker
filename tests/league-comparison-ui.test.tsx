import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LeagueComparisonTable } from "../app/ui/recommendation/_components/LeagueComparisonTable";
import { LeagueResultState } from "../app/ui/recommendation/_components/LeagueComparisonPanel";
import type { LeagueComparison } from "../app/domain/league-comparison";

vi.mock("../lib/supabase/client", () => ({ createClient: vi.fn() }));
const report = (): LeagueComparison => ({
  status: "ok", filters: { season: "2024-25", team: "TOR", side: "offense", season_phase: "regular_season" },
  selected_team_in_league: true, dataset_version: "fixture-v1", aggregation_version: "regular-season",
  rows: [{ play_type: "Cut",
    team: { possessions: 10, field_goal_attempts: 8, fg_pct: 0.5, efg_pct: 0.6, tov_pct: 0.2, ft_pct: 0.1 },
    league: { possessions: 100, field_goal_attempts: 80, fg_pct: 0.6, efg_pct: 0.6, tov_pct: 0.1, ft_pct: 0.2 },
    difference_pp: { fg_pct: -10, efg_pct: 0, tov_pct: 10, ft_pct: -10 } }],
});

describe("league comparison table", () => {
  it("shows four rates, matching denominators and signed percentage points", () => {
    const html = renderToStaticMarkup(<LeagueComparisonTable data={report()} />);
    for (const text of ["Cut", "FG%", "eFG%", "Turnover possessions", "Free-throw possessions", "Attempts (FGA)",
      "50.0%", "60.0%", "-10.0 pp", "+10.0 pp", "0.0 pp", "The league includes TOR", "Small sample", "fixture-v1"])
      expect(html).toContain(text);
    expect(html).toContain('rowSpan="3"');
    expect(html).toContain("not free-throw shooting accuracy");
    expect(html).toContain("rounded source rates");
  });
  it("changes only warnings when the cutoff changes", () => {
    const html = renderToStaticMarkup(<LeagueComparisonTable data={report()} minimum={1} />);
    expect(html).not.toContain('>Small sample</span>');
    expect(html).toContain("50.0%"); expect(html).toContain("-10.0 pp");
  });
  it("preserves zero attempts without showing zero shooting rates", () => {
    const data = report(), row = data.rows[0];
    row.team!.field_goal_attempts = 0; row.team!.fg_pct = row.team!.efg_pct = null;
    row.difference_pp.fg_pct = row.difference_pp.efg_pct = null;
    const html = renderToStaticMarkup(<LeagueComparisonTable data={data} />);
    expect(html).toContain('<td>0</td><td>—</td><td>—</td><td>20.0%</td><td>10.0%</td>');
  });
  it("labels defense as allowed rates and missing seasons as unavailable", () => {
    const data = report(); data.filters.side = "defense";
    expect(renderToStaticMarkup(<LeagueComparisonTable data={data} />)).toContain("Rates allowed by the defense");
    data.status = "no_data"; data.rows = [];
    const html = renderToStaticMarkup(<LeagueComparisonTable data={data} />);
    expect(html).toContain("No recorded defense data for TOR in 2024-25");
    expect(html).not.toContain("<table");
  });
  it("hides results during loading or failure and provides retry", () => {
    const state = { data: report(), loading: true, error: "" };
    const loading = renderToStaticMarkup(<LeagueResultState state={state} minimum={20} retry={() => {}} />);
    expect(loading).toContain("Loading league comparison"); expect(loading).not.toContain("<table");
    const failed = renderToStaticMarkup(<LeagueResultState state={{ ...state, loading: false, error: "Service unavailable" }} minimum={20} retry={() => {}} />);
    expect(failed).toContain('role="alert"'); expect(failed).toContain("Retry comparison");
    expect(failed).not.toContain("<table");
  });
});

describe("league readouts", () => {
  const mainView = (data: LeagueComparison, metric: "fg_pct" | "efg_pct" | "tov_pct" | "ft_pct", minimum = 1) =>
    renderToStaticMarkup(<LeagueComparisonTable data={data} minimum={minimum} initialMetric={metric} />).split("<details")[0];

  it("shows one metric and one row per play type before the full-data disclosure", () => {
    const html = mainView(report(), "fg_pct");
    expect(html.match(/<tr[ >]/g)).toHaveLength(2);
    expect(html).toContain('data-label="TOR"'); expect(html).toContain('data-label="League"');
    expect(html).toContain("50.0%"); expect(html).toContain("60.0%");
    expect(html).toContain("10.0 pp below league");
    expect(html).not.toContain("20.0%");
    expect(html).toContain("8 shot attempts");
    expect(html).toContain('style="width:50%"');
    expect(html).toContain('style="width:60%"');
  });

  it("changes the basketball interpretation for defense without changing values", () => {
    const data = report();
    const offense = mainView(data, "fg_pct");
    expect(offense).toContain("Makes shots less often");
    expect(offense).toContain('data-tone="caution"');
    data.filters.side = "defense";
    const defense = mainView(data, "fg_pct");
    expect(defense).toContain("Opponents make shots less often");
    expect(defense).toContain('data-tone="favorable"');
    expect(defense).toContain("50.0%"); expect(defense).toContain("10.0 pp below league");
  });

  it("treats turnovers as worse on offense and better when forced on defense", () => {
    const data = report();
    expect(mainView(data, "tov_pct")).toContain("Turns the ball over more often");
    expect(mainView(data, "tov_pct")).toContain('data-tone="caution"');
    data.filters.side = "defense";
    const html = mainView(data, "tov_pct");
    expect(html).toContain("Forces more turnovers");
    expect(html).toContain('data-tone="favorable"');
    expect(html).toContain("20.0%"); expect(html).toContain("10 possessions");
  });

  it("uses the selected rate denominator for warnings and the headline", () => {
    const data = report(); data.rows[0].team!.possessions = 25;
    const shooting = mainView(data, "fg_pct", 20);
    expect(shooting).toContain("Small sample");
    expect(shooting).toContain("More data is needed");
    expect(shooting).toContain("50.0%");
    const turnovers = mainView(data, "tov_pct", 20);
    expect(turnovers).not.toContain('>Small sample</span>');
    expect(turnovers).toContain("Cut: turns the ball over more often");
    expect(turnovers).toContain("25 possessions");
  });

  it("keeps eFG distinct from shots made and scales its bars up to 150 percent", () => {
    const data = report();
    data.rows[0].team!.efg_pct = 1.2; data.rows[0].league!.efg_pct = 0.6;
    data.rows[0].difference_pp.efg_pct = 60;
    const html = mainView(data, "efg_pct");
    expect(html).toContain("120.0%"); expect(html).toContain("60.0 pp above league");
    expect(html).toContain('style="width:80%"'); expect(html).toContain('style="width:40%"');
    expect(html).toContain("Gets more scoring value from shots");
    expect(html).toContain("extra value of three-pointers");
  });

  it("explains free-throw possession rates without confusing them with shooting accuracy", () => {
    const data = report();
    expect(mainView(data, "ft_pct")).toContain("Gets free throws less often");
    expect(mainView(data, "ft_pct")).toContain("not free-throw shooting accuracy");
    data.filters.side = "defense";
    const html = mainView(data, "ft_pct");
    expect(html).toContain("Allows fewer free-throw possessions");
    expect(html).toContain('data-tone="favorable"');
  });

  it("does not pick a missing rate for the headline or convert it to zero", () => {
    const data = report(); data.rows[0].team = null; data.rows[0].difference_pp.fg_pct = null;
    const html = mainView(data, "fg_pct");
    expect(html).toContain("Comparison unavailable");
    expect(html).toContain("More data is needed");
    expect(html).not.toContain("<strong>0.0%</strong>");
    expect(html).toContain("<small>Unavailable</small>");
    expect(html.match(/aria-hidden="true"/g)).toHaveLength(1);
    expect(html).not.toContain('style="width:0%"');
  });

  it("updates the season, team and values from the selected report", () => {
    const data = report(); data.filters.team = "BOS"; data.filters.season = "2026-27";
    data.rows[0].team!.fg_pct = 0.8; data.rows[0].difference_pp.fg_pct = 20;
    const html = mainView(data, "fg_pct");
    expect(html).toContain("How does BOS compare?"); expect(html).toContain("2026-27");
    expect(html).toContain("80.0%"); expect(html).toContain("20.0 pp above league");
    expect(html).not.toContain("TOR"); expect(html).not.toContain("2024-25");
  });
});
