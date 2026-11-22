import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
import { BaselineEmptyState } from "../app/ui/recommendation/_components/MatchupPageClient";
import { ContextResults } from "../app/ui/recommendation/_components/ContextPageClient";
import { GameplanEmptyShots } from "../app/ui/gameplan/_components/GameplanClient";

describe("Completed coach requests with no matching rows", () => {
  it("distinguishes an empty baseline response from an unrun selection", () => {
    const html = renderToStaticMarkup(<BaselineEmptyState completed hasRows={false} />);
    expect(html).toContain("No play-type data is available for this matchup and season.");
    expect(html).not.toContain("click Run Baseline");
  });
  it("does not show prior Context results or ask to run an already completed empty response", () => {
    const html = renderToStaticMarkup(<ContextResults current loading={false} hasRows={false} matchupLabel="TOR vs BOS • 2019-20"><p>Old ranking: Cut 1.237</p></ContextResults>);
    expect(html).toContain("No play-type data is available for this matchup and season.");
    expect(html).not.toContain("Old ranking");
    expect(html).not.toContain("click Run Context");
  });
  it("explains absent shot data independently from the completed play-type rankings", () => {
    const html = renderToStaticMarkup(<GameplanEmptyShots completed error={null} />);
    expect(html).toContain("No shot data is available for this matchup and season.");
    expect(html).not.toContain("Build a gameplan to see");
  });
  it("keeps a service failure separate from a successful empty response", () => {
    const html = renderToStaticMarkup(<GameplanEmptyShots completed error="Service unavailable" />);
    expect(html).toContain("could not load");
    expect(html).not.toContain("No shot data");
  });
});
