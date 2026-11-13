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
