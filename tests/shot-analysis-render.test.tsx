import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { normalizeShotAnalysis } from "../app/domain/shot-analysis";
vi.mock("../app/services/shotAnalysis", () => ({ fetchShotMlAnalysis: vi.fn() }));
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
