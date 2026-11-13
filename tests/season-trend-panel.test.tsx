import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import SeasonTrendPanel, { TrendResultState } from "../app/ui/recommendation/_components/SeasonTrendPanel";
import type { SeasonTrend } from "../app/domain/season-trend";

vi.mock("../lib/supabase/client", () => ({ createClient: vi.fn() }));
const report = (): SeasonTrend => ({ status: "ok",
  filters: { team: "TOR", play_type: "Cut", side: "offense", season_phase: "regular_season" },
  dataset_version: "fixture", aggregation_version: "regular-season", rows: [
    { season: "2024-25", possessions: 10, points: 12, team_possessions: 100, usage_pct: 10, ppp: 1.2 }],
});
const render = (state: { data: SeasonTrend | null; loading: boolean; error: string }, minimum = 20) =>
  renderToStaticMarkup(<TrendResultState state={state} minimum={minimum} retry={() => {}} />);

describe("season trend states", () => {
  it("hides old values while loading or showing a retryable error", () => {
    const pending = render({ data: report(), loading: true, error: "" });
    expect(pending).toContain("Loading season trend"); expect(pending).not.toContain("<table");
    const failed = render({ data: report(), loading: false, error: "Service unavailable" });
    expect(failed).toContain('role="alert"'); expect(failed).toContain("Retry trend");
    expect(failed).not.toContain("<table"); expect(failed).not.toContain("<svg");
  });
  it("shows one metric at a time with the matching table and source details", () => {
    const html = render({ data: report(), loading: false, error: "" });
    expect(html).toContain('role="group" aria-label="Trend metric"');
    expect(html).toContain('aria-pressed="true">PPP'); expect(html).toContain('aria-pressed="false">Usage');
    expect(html.match(/<svg /g)).toHaveLength(1); expect(html.match(/<table>/g)).toHaveLength(1);
    for (const text of ["1.200", "10.0%", "How this trend is calculated", "play-type points", "fixture", "not describe recent-game form"])
      expect(html).toContain(text);
  });
  it("keeps unavailable calendar rows visible in a no-data result", () => {
    const data = report(); data.status = "no_data";
    data.rows[0] = { season: "2024-25", possessions: null, points: null, team_possessions: 100, usage_pct: null, ppp: null };
    const html = render({ data, loading: false, error: "" });
    expect(html).toContain("No recorded possessions"); expect(html).toContain("Not recorded");
    expect(html).toContain("2024-25"); expect(html).not.toContain("<td>0.000</td>"); expect(html).not.toContain("<circle");
  });
  it("changes the small-sample warning without filtering results", () => {
    const html = render({ data: report(), loading: false, error: "" }, 5);
    expect(html).toContain("fewer than 5 possessions"); expect(html).not.toContain('>Small sample</span>');
    expect(html).toContain("1.200"); expect(html).toContain("10.0%");
  });
  it("asks for an incomplete selection instead of starting a request", () => {
    const html = renderToStaticMarkup(<SeasonTrendPanel filters={{ team: "TOR", play_type: "" }} />);
    expect(html).toContain("Select a team and play type"); expect(html).not.toContain("Loading");
  });
});
