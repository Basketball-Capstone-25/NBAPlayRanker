import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import example from "./fixtures/playtype-dashboard.json";
import type { PlaytypeDashboard } from "../app/domain/playtype-dashboard";
import { OUTCOME_METRICS, type OutcomeKey, type OutcomeRate, type PossessionOutcomes } from "../app/domain/possession-outcomes";
import UsageReadout from "../app/ui/recommendation/_components/UsageReadout";
import PossessionScenario from "../app/ui/recommendation/_components/PossessionScenario";
import { PossessionOutcomesTable } from "../app/ui/recommendation/_components/PossessionOutcomesTable";

const report = () => structuredClone(example) as PlaytypeDashboard;
const usage = (data: PlaytypeDashboard) => renderToStaticMarkup(<UsageReadout data={data} />);
const focus = (type: string) => {
  const data = report();
  data.usage_efficiency.rows = data.usage_efficiency.rows.filter(row => row.play_type === type);
  return data;
};
const outcomes = (): PossessionOutcomes => ({
  status: "ok", filters: { team: "TOR", season: "2024-25", side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture", aggregation_version: "regular-season",
  rows: [{ play_type: "Cut", possessions: 10,
    rates: Object.fromEntries(OUTCOME_METRICS.map(({ key }, index) => [key, {
      value: [0.7, 0, 0.3, 0.2, 0.1][index], status: "ok", usable_possessions: 10, missing_possessions: 0, invalid_possessions: 0,
    }])) as Record<OutcomeKey, OutcomeRate> }],
});

describe("basketball usage readout", () => {
  it("keeps the team, league and opponent comparisons distinct", () => {
    const data = report(), before = structuredClone(data);
    const html = usage(data);
    expect(html).toContain('value="PRRollMan" selected=""');
    expect(html).toContain("113.5"); expect(html).toContain("110.4");
    expect(html).toContain("3.1 more points per 100 of these possessions");
    expect(html).toContain("BOS allowed 97.1 points");
    expect(html).toContain("102.9 for the league");
    expect(html).toContain("separate season profiles");
    expect(html).not.toContain("both make this worth reviewing");
    expect(data).toEqual(before);
  });
  it("does not turn absent opponent defense into a zero defensive rate", () => {
    const html = usage(focus("Cut"));
    expect(html).toContain("11.0 fewer points per 100");
    expect(html).toContain("no opponent defensive rate for this play");
    expect(html).not.toContain("BOS allowed 0");
  });
  it("withholds a strong takeaway for small offensive or defensive samples", () => {
    const offense = focus("Handoff"); offense.usage_efficiency.rows[0].sample_ok = false;
    const html = usage(offense);
    expect(html).toContain("Too few possessions for a strong takeaway");
    expect(html).not.toContain("12.2 fewer points");
    expect(html).not.toContain("BOS allowed");
    const defense = focus("Handoff"); defense.opponent_defense.rows[0].sample_ok = false;
    expect(usage(defense)).toContain("opponent’s sample is too small");
  });
  it("describes tiny scoring differences as close rather than assigning an inflated advantage", () => {
    const data = focus("PRRollMan"), row = data.usage_efficiency.rows[0];
    row.ppp = row.league_ppp + 0.0004; row.ppp_delta = 0.0004;
    const html = usage(data);
    expect(html).toContain("Scoring is close to the league rate");
    expect(html).not.toContain("0.0 more points");
  });
  it("only points to a joint film-review opportunity when both supported rates align", () => {
    const data = focus("PRRollMan"), defense = data.opponent_defense.rows.find(row => row.play_type === "PRRollMan")!;
    defense.allowed_ppp = 1.2; defense.league_allowed_ppp = 1; defense.allowed_ppp_delta = 0.2;
    const html = usage(data);
    expect(html).toContain("both make this worth reviewing on film");
    expect(html).toContain("not what will happen when these teams play");
  });
  it("shows no selected play or invented numbers for an empty result", () => {
    const data = report(); data.usage_efficiency.rows = [];
    expect(usage(data)).toBe("");
  });
});

describe("possession scenario entry", () => {
  it("waits for two choices and clearly labels the fixed-rate assumption", () => {
    const html = renderToStaticMarkup(<PossessionScenario rows={report().usage_efficiency.rows} />);
    expect(html).toContain("Use less"); expect(html).toContain("Use more");
    expect(html).toContain("fixed-rate illustration, not a prediction or points per game");
    expect(html).not.toContain("<output"); expect(html).not.toContain('type="range"');
  });
  it("does not offer a scenario without two supported play types", () => {
    const rows = report().usage_efficiency.rows.map((row, index) => ({ ...row, sample_ok: index === 0 }));
    const html = renderToStaticMarkup(<PossessionScenario rows={rows} />);
    expect(html).toContain("Two play types must meet the sample cutoff");
    expect(html).not.toContain("<select"); expect(html).not.toContain("<output");
  });
});

describe("possession outcome readout", () => {
  it("uses independent bars, approximate language and a small-sample warning", () => {
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={outcomes()} />);
    expect(html).toContain("Small sample — treat these rates cautiously");
    expect(html).toContain("About 70.0 out of every 100 produced points");
    expect(html).toContain("About 0.0 ended in a turnover");
    expect(html).toContain('style="width:70%"'); expect(html).toContain('style="width:0%"');
    expect(html).toContain("Rates can overlap. Do not add these percentages");
    expect(html).toContain("Each bar has its own 0–100% scale");
  });
  it("keeps invalid and missing rates unavailable in both the text and bars", () => {
    const data = outcomes();
    data.rows[0].rates.scoring = { value: null, status: "invalid", usable_possessions: 9, missing_possessions: 0, invalid_possessions: 1 };
    data.rows[0].rates.turnover = { value: null, status: "missing", usable_possessions: 9, missing_possessions: 1, invalid_possessions: 0 };
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={data} />);
    expect(html).toContain("scoring rate is unavailable"); expect(html).toContain("turnover rate is unavailable");
    expect(html).toContain("Invalid source");
    expect(html).not.toContain('style="width:70%"'); expect(html).not.toContain('style="width:0%"');
    expect(html).toContain('style="width:30%"');
  });
  it("does not fabricate a play, number or chart when no source rows exist", () => {
    const data = outcomes(); data.rows = []; data.status = "no_data";
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={data} />);
    expect(html).toContain("No recorded offensive data for TOR in 2024-25");
    expect(html).not.toContain("<select"); expect(html).not.toContain('style="width:');
    expect(html).not.toContain("<table");
  });
});
