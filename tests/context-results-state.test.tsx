import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
vi.mock("../lib/supabase/client", () => ({ createClient: () => ({}) }));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));
import { ContextResults } from "../app/ui/recommendation/_components/ContextPageClient";

const render = (current: boolean, loading = false, hasRows = true) => renderToStaticMarkup(
  <ContextResults current={current} loading={loading} hasRows={hasRows} matchupLabel={current ? "TOR vs BOS • 2024-25" : "LAL vs DEN • 2024-25"}>
    <p>Previous TOR ranking: Cut 1.237</p><table><tbody><tr><td>Previous breakdown 0.047</td></tr></tbody></table>
  </ContextResults>,
);

describe("Context results after controls change", () => {
  it("hides the old ranking and breakdown instead of relabelling them as the new matchup", () => {
    const html = render(false);
    expect(html).not.toContain("Previous TOR ranking");
    expect(html).not.toContain("Previous breakdown");
    expect(html).not.toContain("LAL vs DEN");
    expect(html).toContain("Inputs changed. Run Context + ML again.");
  });
  it("keeps completed current results and their heading visible", () => {
    const html = render(true);
    expect(html).toContain("Previous TOR ranking");
    expect(html).toContain("Previous breakdown");
    expect(html).toContain("TOR vs BOS");
  });
  it("suppresses result content and matchup heading while loading", () => {
    const html = render(false, true);
    expect(html).toContain("Generating ML + context ranking");
    expect(html).not.toContain("Previous TOR ranking");
    expect(html).not.toContain("LAL vs DEN");
  });
});
