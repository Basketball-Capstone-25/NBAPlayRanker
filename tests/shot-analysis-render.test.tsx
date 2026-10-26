import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { normalizeShotAnalysis } from "../app/domain/shot-analysis";
vi.mock("../app/services/shotAnalysis", () => ({ fetchShotMlAnalysis: vi.fn(), fetchShotModelMetrics: vi.fn() }));
import { ShotAnalysisResults } from "../app/ui/shot-analysis/_components/ShotStatisticalAnalysisPageClient";

const current = {
  dataset: { rows_used: 22, n_seasons: 2, feature_spec: { numeric: ["DIST", "MARGIN"], categorical: ["TEAM_ABBR"], target: "POINTS" } },
  eda: { points: { hist: { bins: [0, 1, 2], counts: [8, 14] } }, missing_counts: { MARGIN: 22 } },
  feature_target_corr: [{ feature: "DIST", corr: -.25, abs: .25 }],
  feature_selection: { method: "SelectKBest(f_regression) on numeric features", k: 1, scores: [{ feature: "DIST", score: 5 }] },
};

describe("shot statistical analysis response", () => {
  it("renders the current backend schema without inventing games, RFE or tuned metrics", () => {
    const html = renderToStaticMarkup(<ShotAnalysisResults data={normalizeShotAnalysis(current)} />);
    expect(html).toContain("22 sampled shot attempts");
    expect(html).toContain("Unavailable games");
    expect(html).toContain("-0.25");
    expect(html).toContain("SelectKBest");
    expect(html).toContain("not included in this analysis");
    expect(html).not.toContain("Best RMSE");
  });
  it("unwraps prepared payloads and shows unavailable optional sections", () => {
    const data = normalizeShotAnalysis({ payload: { dataset: { rows_used: 10 } } });
    const html = renderToStaticMarkup(<ShotAnalysisResults data={data} />);
    expect(html).toContain("10 sampled shot attempts");
    expect(html).toContain("Feature scores are unavailable");
    expect(html).toContain("Input correlations are unavailable");
  });
  it("never turns null statistics into measured zero values", () => {
    const data = normalizeShotAnalysis({ ...current, feature_target_corr: [{ feature: "MARGIN", corr: null, abs: null }] });
    expect(data.feature_target_corr[0].corr).toBeNull();
    expect(renderToStaticMarkup(<ShotAnalysisResults data={data} />)).toContain("Unavailable");
  });
});

import { ShotLaterSeasonResults } from "../app/ui/shot-analysis/_components/ShotLaterSeasonResults";
it("labels the reserved later season separately from training cross-validation", () => {
  const html = renderToStaticMarkup(<ShotLaterSeasonResults selectedModel="gbr" result={{ train_seasons: ["2023-24"], test_season: "2024-25", train_rows: 200, test_rows: 50, metrics: [{ model: "training_mean", RMSE: 1.2, MAE: 1.1, R2: null }], limitation: "This does not measure pre-possession forecasts." }} />);
  expect(html).toContain("excluded from model selection");
  expect(html).toContain("Training-mean baseline");
  expect(html).toContain("1.2000");
  expect(html).toContain("Unavailable");
});


import { ShotMetricsContent } from "../app/ui/shot-analysis/_components/ShotModelMetricsPageClient";
const savedMetrics = {
  n_splits: 5, source_sha256: "a".repeat(64), best_model: "gbr",
  metrics: [{ model:"gbr",RMSE_mean:1.16,RMSE_std:.01,MAE_mean:1.08,MAE_std:.02,R2_mean:.04,R2_std:.01 }],
};

describe("saved shot metrics presentation", () => {
  it("uses readable model names and removes controls for unsupported evaluations", () => {
    const html = renderToStaticMarkup(<ShotMetricsContent data={savedMetrics} loading={false} error={null} onReload={() => {}} />);
    expect(html).toContain("Gradient boosting");
    expect(html).toContain("Training-season cross-validation");
    expect(html).toContain("Export training results (CSV)");
    expect(html).toContain(`href="/api/exports/shot-metrics?version=${savedMetrics.source_sha256}"`);
    expect(html).toContain('target="_blank" rel="noopener noreferrer"');
    expect(html).not.toContain(' download=');
    expect(html).not.toContain("Holdout comparison");
    expect(html).not.toContain("<select");
    expect(html).not.toContain("Auto-refresh");
    expect(html).toContain("<details><summary>Evaluation details</summary>");
    const beforeDetails = html.split("<details>")[0];
    expect(beforeDetails).toContain("do not measure pre-possession forecasts");
    expect(beforeDetails).not.toContain("sha256:");
  });
  it("does not offer a metrics download without a matching source or supported folds", () => {
    for (const data of [{ ...savedMetrics, source_sha256: undefined }, { ...savedMetrics, n_splits: 3 }]) {
      const html = renderToStaticMarkup(<ShotMetricsContent data={data} loading={false} error={null} onReload={() => {}} />);
      expect(html).not.toContain("Export training results");
    }
  });
  it("hides old results and download actions during loading and after failure", () => {
    for (const state of [{loading:true,error:null},{loading:false,error:"Saved results could not be loaded."}]) {
      const html = renderToStaticMarkup(<ShotMetricsContent data={savedMetrics} {...state} onReload={() => {}} />);
      expect(html).not.toContain("Export training results");
      expect(html).not.toContain("Gradient boosting");
      expect(html).not.toContain("sha256:");
      expect(html).toContain(state.loading ? "Loading saved results" : "Results unavailable");
    }
  });
  it("keeps shot sample and scope visible while statistical tables start collapsed", () => {
    const html = renderToStaticMarkup(<ShotAnalysisResults data={normalizeShotAnalysis(current)} />);
    const beforeDetails = html.split('<details')[0];
    expect(beforeDetails).toContain("22 sampled shot attempts");
    expect(beforeDetails).toContain("not a pre-possession forecast");
    expect(beforeDetails).toContain("Points distribution");
    expect(beforeDetails).not.toContain("SelectKBest");
    expect(html).toContain('<summary>Analysis details</summary>');
    expect(html).not.toContain('<details open');
  });
});
