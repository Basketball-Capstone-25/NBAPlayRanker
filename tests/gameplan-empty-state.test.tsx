import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));
import Gameplan from "../app/ui/gameplan/_components/GameplanClient";

describe("Gameplan before a matchup is built", () => {
  it("does not export shot advice before a completed gameplan", () => {
    const html = renderToStaticMarkup(<Gameplan />);
    const pdfButton = html.match(/<button[^>]*>Export PDF<\/button>/)?.[0];
    expect(pdfButton).toBeDefined();
    expect(pdfButton).toContain('disabled=""');
  });

  it("shows a prompt instead of unearned shot advice", () => {
    const html = renderToStaticMarkup(<Gameplan />);
    expect(html).toContain("Build a gameplan to see shot priorities for this matchup.");
    expect(html).not.toContain("We want a high-quality shot");
    expect(html).not.toContain("Primary: get a clean rim touch");
    expect(html).not.toContain("Dataset2 endpoint");
  });
});
