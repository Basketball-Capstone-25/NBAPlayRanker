import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import DatasetReloadPanel, { DatasetReloadView, type ReloadState } from "../app/ui/data-explorer/DatasetReloadPanel";
import type { DatasetReloadResult } from "../app/domain/dataset-reload";

vi.mock("../lib/supabase/client", () => ({ createClient: () => ({ auth: { getSession: vi.fn() } }) }));
vi.mock("../app/infrastructure/dataset-reload", () => ({ reloadDataset: vi.fn() }));

const result: DatasetReloadResult = { dataset: "synergy", status: "reloaded", source_file: "synergy.csv",
  previous_version: "fixture-old", dataset_version: "fixture-new", changed: true, rows: 47982,
  seasons: ["2023-24", "2024-25"], quality_status: "pass" };
const render = (state: ReloadState) => renderToStaticMarkup(<DatasetReloadView state={state} onReload={() => {}} />);

describe("dataset reload controls", () => {
  it("offers separate stored-file actions without starting a reload on render", () => {
    const html = renderToStaticMarkup(<DatasetReloadPanel onReloaded={() => {}} />);
    expect(html).toContain("Reload play types");
    expect(html).toContain("Reload shots");
    expect(html).toContain("does not download new NBA data");
    expect(html).not.toContain('disabled=""');
    expect(html).not.toContain("Reloaded");
  });

  it.each(["synergy", "pbp"] as const)("disables both actions while %s is reloading", dataset => {
    const html = render({ kind: "loading", dataset });
    expect(html.match(/disabled=""/g)).toHaveLength(2);
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('role="status"');
    expect(html).not.toContain("100%");
    expect(html).not.toContain("Reloaded");
  });

  it("reports changed data and keeps detailed versions collapsed", () => {
    const html = render({ kind: "ready", dataset: "synergy", result });
    for (const text of ["A different file version", "47,982 source rows", "2 seasons", "Source checks passed",
      "fixture-old", "fixture-new", "synergy.csv"]) expect(html).toContain(text);
    expect(html).toContain("<details>");
    expect(html).not.toContain("<details open");
  });

  it("treats an unchanged reload as success without claiming shot quality passed", () => {
    const shots: DatasetReloadResult = { ...result, dataset: "pbp_shots", source_file: "shots_clean.parquet",
      previous_version: "fixture-new", changed: false, quality_status: undefined };
    const html = render({ kind: "ready", dataset: "pbp", result: shots });
    expect(html).toContain("Reloaded shots");
    expect(html).toContain("The saved file is unchanged");
    expect(html).not.toContain("Source checks passed");
    expect(html).not.toContain('role="alert"');
  });

  it("allows a new attempt after a confirmed rejection and does not invent a success", () => {
    const html = render({ kind: "error", dataset: "synergy", uncertain: false, message: "The previous data is still in use." });
    expect(html).toContain('role="alert"');
    expect(html).toContain("previous data is still in use");
    expect(html).not.toContain('disabled=""');
    expect(html).not.toContain("Reloaded");
  });

  it("requires checking the page after an unknown outcome instead of repeating the POST", () => {
    const html = render({ kind: "error", dataset: "pbp", uncertain: true, message: "The reload result could not be confirmed." });
    expect(html.match(/disabled=""/g)).toHaveLength(2);
    expect(html).toContain("Refresh page");
    expect(html).toContain("before another reload");
    expect(html).not.toContain("Reloaded");
  });
});
