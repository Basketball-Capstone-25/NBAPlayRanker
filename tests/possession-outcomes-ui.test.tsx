import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OUTCOME_METRICS, type OutcomeKey, type OutcomeRate, type PossessionOutcomes } from "../app/domain/possession-outcomes";
import { PossessionOutcomesTable } from "../app/ui/recommendation/_components/PossessionOutcomesTable";
import PossessionOutcomesPanel, { PossessionOutcomesResultState } from "../app/ui/recommendation/_components/PossessionOutcomesPanel";

vi.mock("../app/infrastructure/possession-outcomes", () => ({ fetchPossessionOutcomes: vi.fn() }));
const report = (): PossessionOutcomes => ({
  status: "ok", filters: { season: "2019-20", team: "PHX", side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture-v1", aggregation_version: "regular-season",
  rows: [{ play_type: "Misc", possessions: 486,
    rates: Object.fromEntries(OUTCOME_METRICS.map(({ key }, index) => [key, {
      value: [0.3476419753, 0.5287901235, 0.2736893004, 0.03078806584, 0.006189300412][index],
      status: "ok", usable_possessions: 486, missing_possessions: 0, invalid_possessions: 0,
    }])) as Record<OutcomeKey, OutcomeRate> }],
});

describe("independent possession outcome table", () => {
  it("shows all five rates without forcing their sum to 100%", () => {
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={report()} />);
    for (const text of ["PHX possession outcomes", "2019-20 regular season", "486", "34.8%", "52.9%", "27.4%", "3.1%", "0.6%",
      "Rates can overlap. Do not add these percentages.", "Approximate rates from rounded source data.", "fixture-v1"])
      expect(html).toContain(text);
    expect(html).toContain('scope="row"'); expect(html).toContain('<tbody><tr>');
    expect(html).not.toContain('<svg'); expect(html).not.toContain('rowSpan=');
    for (const metric of OUTCOME_METRICS) expect(html).toContain(`<th scope="col">${metric.label}</th>`);
  });
  it("keeps definitions collapsed and explains weighting and source limitations", () => {
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={report()} />);
    expect(html).toContain("Rate definitions and calculations"); expect(html).not.toContain(' open=');
    for (const metric of OUTCOME_METRICS) { expect(html).toContain(metric.source); expect(html).toContain(metric.definition); }
    expect(html).toContain("Σ(player possessions × source rate) ÷ total play-type possessions");
    expect(html).toContain("not free-throw shooting accuracy");
    expect(html).toContain("do not recover exact event counts");
  });
  it("shows invalid data visibly and makes each metric's coverage accessible", () => {
    const data = report();
    data.rows[0].rates.scoring = { value: null, status: "invalid", usable_possessions: 400, missing_possessions: 6, invalid_possessions: 80 };
    data.rows[0].rates.free_throw = { value: null, status: "missing", usable_possessions: 400, missing_possessions: 86, invalid_possessions: 0 };
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={data} />);
    expect(html).toContain('>Invalid source</td>'); expect(html).toContain('>—</td>');
    expect(html).toContain('aria-label="Scoring: Invalid source. 400 of 486 possessions have a usable rate; 6 missing; 80 invalid."');
    expect(html).toContain('aria-label="Free throws: —. 400 of 486 possessions have a usable rate; 86 missing; 0 invalid."');
    expect(html).toContain('>52.9%</td>'); expect(html).not.toContain('>34.8%</td>');
  });
  it("distinguishes a genuine zero rate from unavailable zero-possession rates", () => {
    const data = report(); data.rows[0].rates.turnover.value = 0;
    expect(renderToStaticMarkup(<PossessionOutcomesTable data={data} />)).toContain('>0.0%</td>');
    data.rows[0].possessions = 0;
    for (const key of OUTCOME_METRICS.map(metric => metric.key)) data.rows[0].rates[key] = {
      value: null, status: "no_possessions", usable_possessions: 0, missing_possessions: 0, invalid_possessions: 0 };
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={data} />);
    expect(html.match(/>—<\/td>/g)).toHaveLength(5); expect(html).not.toContain('>0.0%</td>');
  });
  it("changes small-sample warnings without hiding rows or changing values", () => {
    const data = report(); data.rows[0].possessions = 11;
    for (const rate of Object.values(data.rows[0].rates)) rate.usable_possessions = 11;
    const warned = renderToStaticMarkup(<PossessionOutcomesTable data={data} minimum={20} />);
    const unflagged = renderToStaticMarkup(<PossessionOutcomesTable data={data} minimum={1} />);
    expect(warned).toContain('>Small sample</span>'); expect(unflagged).not.toContain('>Small sample</span>');
    for (const html of [warned, unflagged]) { expect(html).toContain('<td>11</td>'); expect(html).toContain('>34.8%</td>'); }
  });
  it("shows an empty selection clearly without a fabricated table", () => {
    const data = report(); data.status = "no_data"; data.rows = [];
    const html = renderToStaticMarkup(<PossessionOutcomesTable data={data} />);
    expect(html).toContain("No recorded offensive data for PHX in 2019-20"); expect(html).not.toContain("<table");
  });
  it("hides prior results during loading or errors and offers retry", () => {
    const state = { data: report(), loading: true, error: "" };
    const loading = renderToStaticMarkup(<PossessionOutcomesResultState state={state} minimum={20} retry={() => {}} />);
    expect(loading).toContain("Loading possession rates"); expect(loading).not.toContain("<table");
    const failed = renderToStaticMarkup(<PossessionOutcomesResultState state={{ ...state, loading: false, error: "Service unavailable" }} minimum={20} retry={() => {}} />);
    expect(failed).toContain('role="alert"'); expect(failed).toContain("Retry rates"); expect(failed).not.toContain("<table");
  });
  it("requires a team and season before starting the request", () => {
    expect(renderToStaticMarkup(<PossessionOutcomesPanel filters={{ season: "2019-20", team: "" }} minimum={20} />))
      .toContain("Select a team and a season");
    expect(renderToStaticMarkup(<PossessionOutcomesPanel filters={{ season: "", team: "PHX" }} minimum={20} />))
      .toContain("Select a team and a season");
    expect(renderToStaticMarkup(<PossessionOutcomesPanel filters={{ season: "2019-20", team: "PHX" }} minimum={20} />))
      .toContain("Loading possession rates");
  });
});
