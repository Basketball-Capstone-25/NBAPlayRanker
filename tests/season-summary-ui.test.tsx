import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import SeasonSummary, { SeasonSummaryResults, SeasonSummaryState } from "../app/ui/_shared/components/SeasonSummary";
import AnalystSeasonSummary, { analystSummaryFilters } from "../app/ui/data-explorer/AnalystSeasonSummary";
import type { SeasonSummary as SummaryData } from "../app/domain/season-summary";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));

const report: SummaryData = {
  status: "ok", message: null,
  filters: { season: "2024-25", our: "TOR", opp: "BOS", k: 1, w_off: 0.7, w_def: 0.3,
    period: "season", side: "offense", season_phase: "regular_season" },
  dataset_version: "test-source", aggregation_version: "test-aggregation", limitations: [],
  summary: { observed_team_ppp: 1.7, modeled_topk_ppp: 0.9, modeled_uplift_ppp: -0.8,
    team_possessions: 400, selected_possessions: 100, selected_usage_pct: 25, k_returned: 1 },
  play_types: [{ rank: 1, play_type: "Cut", possessions: 100, usage_pct: 25 }],
};

describe("season summary display", () => {
  it("separates modeled and observed values, retaining a negative difference and full-season usage", () => {
    const html = renderToStaticMarkup(<SeasonSummaryResults data={report} />);
    for (const text of ["Observed team PPP", "1.700", "Modeled Top-K PPP", "0.900", "-0.800", "25.0%", "400", "1. Cut", "100 poss.", "2024-25 regular season", "not measured improvement or a forecast"])
      expect(html).toContain(text);
    expect(html).not.toContain("100.0%");
    expect(html).toContain("width:25%");
    expect(html).toContain("Usage = play-type possessions ÷ all recorded offensive play-type possessions × 100");
  });
  it("shows missing data without made-up zero metrics", () => {
    const data: SummaryData = { ...report, status: "no_data", message: "No data for this period.", summary: null, play_types: [] };
    const html = renderToStaticMarkup(<SeasonSummaryResults data={data} />);
    expect(html).toContain("No data for this period");
    expect(html).not.toContain("Observed team PPP");
    expect(html).not.toContain("0.000");
  });
  it("shows loading or retry without displaying previous results", () => {
    const base = { key: "test", data: report, error: null, loading: true };
    const loading = renderToStaticMarkup(<SeasonSummaryState state={base} retry={() => {}} />);
    expect(loading).toContain("Loading season summary");
    expect(loading).not.toContain("1.700");
    const failure = renderToStaticMarkup(<SeasonSummaryState state={{ ...base, loading: false, error: "Service unavailable" }} retry={() => {}} />);
    expect(failure).toContain('role="alert"');
    expect(failure).toContain("Retry summary");
    expect(failure).not.toContain("1.700");
  });
  it("does not load when rankings are unavailable or the matchup is invalid", () => {
    for (const filters of [null, { ...report.filters, opp: "TOR" }, { ...report.filters, w_off: NaN }]) {
      const html = renderToStaticMarkup(<SeasonSummary filters={filters} unavailableReason="Run Baseline first." />);
      expect(html).toContain("Run Baseline first.");
      expect(html).not.toContain("Loading season summary");
    }
  });
});

describe("analyst summary scope", () => {
  it("uses one offensive team and a separate opponent, with a fixed top five", () => {
    expect(analystSummaryFilters("2024-25", "TOR", "offense", "BOS"))
      .toEqual({ season: "2024-25", our: "TOR", opp: "BOS", k: 5, w_off: 0.7 });
    const html = renderToStaticMarkup(<AnalystSeasonSummary season="2024-25" team="TOR" side="offense" teams={["TOR", "BOS"]} />);
    expect(html).toContain("Summary opponent");
    expect(html).toContain('value="BOS" selected=""');
    expect(html).toContain("Table search, play-type filters, possession minimum and row limit do not change it");
    expect(html).toContain("Loading season summary");
  });
  it("requires a single team and offense before requesting data", () => {
    for (const [season, team, side, opponent] of [["", "TOR", "offense", "BOS"], ["2024-25", "", "offense", "BOS"], ["2024-25", "TOR", "defense", "BOS"], ["2024-25", "TOR", "offense", "TOR"]]) {
      expect(analystSummaryFilters(season, team, side, opponent)).toBeNull();
    }
    const html = renderToStaticMarkup(<AnalystSeasonSummary season="2024-25" team="" side="offense" teams={["TOR", "BOS"]} />);
    expect(html).toContain("Select one team and Offense");
    expect(html).not.toContain("Loading season summary");
  });
  it("chooses a different opponent when our team changes to the selected opponent", () => {
    const html = renderToStaticMarkup(<AnalystSeasonSummary season="2024-25" team="BOS" side="offense" teams={["TOR", "BOS"]} />);
    expect(html).toContain('value="TOR" selected=""');
    expect(html).not.toContain('value="BOS"');
  });
});
