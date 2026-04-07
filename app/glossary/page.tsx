// app/glossary/page.tsx
//
// Glossary
//
// What this page does:
// - Plain-English definitions for *everything the product uses* across all pages.
// - Covers Dataset1 (Synergy play types) + Dataset2 (NBA play-by-play shots) + Gameplan.
// - Includes “Why it matters” and “Where you see it” so reviewers can trace terms to screens.
//

"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { CSSProperties } from "react";

type WhereLink = { label: string; href: string };

const PRODUCT_LINKS: WhereLink[] = [
  { label: "Home", href: "/" },
  { label: "Data Explorer (Dataset1)", href: "/data-explorer" },
  { label: "Matchup (Baseline)", href: "/matchup" },
  { label: "AI Context Simulator", href: "/context" },
  { label: "Model Performance", href: "/model-metrics" },
  { label: "Statistical Analysis", href: "/statistical-analysis" },
  { label: "Gameplan", href: "/gameplan" },
  { label: "Shot Plan (Dataset2)", href: "/shot-plan" },
  { label: "Shot Heatmap (Dataset2)", href: "/shot-heatmap" },
  { label: "Shots Explorer (Dataset2)", href: "/shot-explorer" },
  { label: "Shot Model Metrics", href: "/shot-model-metrics" },
  { label: "Shot Statistical Analysis", href: "/shot-statistical-analysis" },
  { label: "Glossary", href: "/glossary" },
];

const TERMS: Array<{
  term: string;
  meaning: string;
  whyItMatters?: string;
  tags?: string[];
  where?: WhereLink[];
  aliases?: string[];
}> = [
  // Product / Pages
  {
    term: "NBA Play Ranker",
    meaning:
      "The overall product: a decision-support tool that ranks play types (Dataset1) and recommends shot priorities (Dataset2), with explainable logic and exports.",
    whyItMatters:
      "Frames the capstone as a complete workflow: trusted baseline → context-aware AI → shot plan + visuals → exports.",
    tags: ["Core", "Pages"],
    where: PRODUCT_LINKS,
    aliases: ["Product", "App"],
  },
  {
    term: "Data Explorer (Dataset1)",
    meaning:
      "A transparency screen that lets you browse the cleaned, team-level play-type table used by the recommenders (filter, preview, export).",
    whyItMatters:
      "It’s the ‘show your work’ proof: reviewers can export the same filtered data and verify calculations outside the app.",
    tags: ["Pages", "Data", "Dataset1"],
    where: [{ label: "Data Explorer", href: "/data-explorer" }],
    aliases: ["Team Playtypes", "team-playtypes"],
  },
  {
    term: "Matchup Console (Baseline)",
    meaning:
      "The explainable recommender screen. Choose season + matchup and get Top-K ranked play types using a transparent formula with shrinkage.",
    whyItMatters:
      "This is the trust anchor: it’s repeatable and easy to defend before showing any AI adjustments.",
    tags: ["Pages", "Models", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["Baseline page", "Baseline"],
  },
  {
    term: "AI Context Simulator",
    meaning:
      "The AI use-case screen. Starts from ML-predicted efficiency and re-ranks play types using game context inputs (score/time) plus an explainable overlay.",
    whyItMatters:
      "Demonstrates practical AI value: the ranking changes in sensible ways depending on situation, and you can explain why.",
    tags: ["Pages", "AI", "Context", "Dataset1"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["Context page", "Context ranking"],
  },
  {
    term: "Model Performance",
    meaning:
      "A defense page that shows holdout evaluation metrics for multiple models (RMSE/MAE/R²), plus tuning evidence pulled from Statistical Analysis.",
    whyItMatters:
      "Shows model choice was validated, not guessed: generalization evidence + tuning/search evidence.",
    tags: ["Pages", "Evaluation", "Models"],
    where: [{ label: "Model Performance", href: "/model-metrics" }],
    aliases: ["Metrics page", "Holdout metrics"],
  },
  {
    term: "Statistical Analysis",
    meaning:
      "A defense page that summarizes EDA, correlations, feature selection, and hyperparameter tuning for Dataset1 ML models.",
    whyItMatters:
      "Supports the ‘method’ story: systematic analysis and selection steps, not ad-hoc choices.",
    tags: ["Pages", "Evaluation", "AI", "Dataset1"],
    where: [{ label: "Statistical Analysis", href: "/statistical-analysis" }],
    aliases: ["EDA page", "Analysis page"],
  },
  {
    term: "Gameplan",
    meaning:
      "A combined workflow screen that ties recommendations together and can interpret natural-language coaching intent into structured context for explanations.",
    whyItMatters:
      "This is the ‘productized’ experience: a single place to go from situation → recommendations → explanation.",
    tags: ["Pages", "Core", "AI"],
    where: [{ label: "Gameplan", href: "/gameplan" }],
    aliases: ["Plan builder"],
  },
  {
    term: "Shot Plan (Dataset2)",
    meaning:
      "A recommendation screen that uses the play-by-play shots dataset to suggest which shot types and zones to target (and optionally who to use).",
    whyItMatters:
      "Adds a second decision layer beyond play types: ‘what shots should we hunt?’ backed by real shot event data.",
    tags: ["Pages", "Dataset2", "Models", "Viz"],
    where: [{ label: "Shot Plan", href: "/shot-plan" }],
    aliases: ["Shot priorities"],
  },
  {
    term: "Shot Heatmap (Dataset2)",
    meaning:
      "A visual screen that renders a shot-location heatmap for a team (and optional opponent / shot type / zone filters).",
    whyItMatters:
      "Turns raw shot data into an instantly understandable spatial picture for coaches and reviewers.",
    tags: ["Pages", "Dataset2", "Viz"],
    where: [{ label: "Shot Heatmap", href: "/shot-heatmap" }],
    aliases: ["Heatmap"],
  },
  {
    term: "Shots Explorer (Dataset2)",
    meaning:
      "A raw data proof page. One row = one shot event from play-by-play. Filter → preview small slice → export CSV for deeper work.",
    whyItMatters:
      "Lets anyone see the underlying events that the Shot Plan and Heatmap are built from (traceability).",
    tags: ["Pages", "Dataset2", "Data", "Exports"],
    where: [{ label: "Shots Explorer", href: "/shot-explorer" }],
    aliases: ["PBP preview", "Shot rows"],
  },
  {
    term: "Shot Model Metrics",
    meaning:
      "A Dataset2 defense page showing evaluation metrics for shot models using grouped holdout splits (to avoid game leakage).",
    whyItMatters:
      "Shows Dataset2 modeling was evaluated properly (not trained and tested on correlated shots from the same game).",
    tags: ["Pages", "Dataset2", "Evaluation", "Models"],
    where: [{ label: "Shot Model Metrics", href: "/shot-model-metrics" }],
    aliases: ["GroupKFold metrics"],
  },
  {
    term: "Shot Statistical Analysis",
    meaning:
      "A Dataset2 analysis page summarizing EDA/correlations/feature selection/model selection for shot modeling.",
    whyItMatters:
      "Completes the defense story for Dataset2 the same way Statistical Analysis does for Dataset1.",
    tags: ["Pages", "Dataset2", "Evaluation", "AI"],
    where: [{ label: "Shot Statistical Analysis", href: "/shot-statistical-analysis" }],
    aliases: ["Shot EDA"],
  },
  // Datasets / Data engineering
  {
    term: "Dataset1 (Synergy Play Types)",
    meaning:
      "The play-type dataset used to compute play-type efficiency (PPP) by team/season, and to power the baseline and context recommendations.",
    whyItMatters:
      "This dataset supports the primary recommendation unit: which *play types* to run versus an opponent.",
    tags: ["Data", "Dataset1", "Core"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
    ],
    aliases: ["Synergy", "Playtype data"],
  },
  {
    term: "Dataset2 (NBA Play-by-Play Shots)",
    meaning:
      "A large play-by-play-derived shot event dataset (one row per shot) used for shot plans, heatmaps, and shot evaluation pages.",
    whyItMatters:
      "Adds a defendable ‘shot selection’ layer and real spatial visuals to the product.",
    tags: ["Data", "Dataset2", "Core"],
    where: [
      { label: "Shot Plan", href: "/shot-plan" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
    aliases: ["PBP", "Shots parquet"],
  },
  {
    term: "Parquet",
    meaning:
      "A columnar data file format designed for analytics. It loads efficiently and supports large datasets.",
    whyItMatters:
      "Dataset2 is large, so parquet helps keep the backend fast and the UI responsive.",
    tags: ["Engineering", "Dataset2", "Data"],
    aliases: ["*.parquet"],
  },
  {
    term: "Meta Options",
    meaning:
      "Dropdown values pulled from the backend (seasons, teams, play types, shot types, zones).",
    whyItMatters:
      "Keeps the UI aligned with whatever data is actually available (and avoids hard-coded lists).",
    tags: ["Engineering", "Data"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
      { label: "Shot Plan", href: "/shot-plan" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
    aliases: ["meta/options"],
  },
  {
    term: "Preview vs Export",
    meaning:
      "Preview = a small slice of rows for fast browsing. Export = a larger CSV designed for Excel/R/Python analysis.",
    whyItMatters:
      "Prevents the browser from freezing while still allowing deep external verification when needed.",
    tags: ["Engineering", "Exports", "Data"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
  },
  {
    term: "Limit",
    meaning:
      "A cap on how many rows the UI requests for a preview table (to keep the page fast).",
    whyItMatters:
      "A practical constraint for committee demos: the app stays snappy even on big data.",
    tags: ["Engineering", "Data"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
    aliases: ["preview limit"],
  },
  {
    term: "min_poss (Minimum possessions filter)",
    meaning:
      "A filter that removes low-sample play-type rows (few possessions) from previews/analysis.",
    whyItMatters:
      "Avoids misleading stats from tiny samples and improves the reliability of conclusions.",
    tags: ["Data", "Stats", "Dataset1"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Statistical Analysis", href: "/statistical-analysis" },
      { label: "Model Performance", href: "/model-metrics" },
    ],
    aliases: ["Min Poss", "Minimum sample"],
  },
  // Core basketball / app inputs
  {
    term: "Season",
    meaning:
      "A specific NBA season label (e.g., 2023-24). Used to filter both datasets.",
    whyItMatters:
      "Prevents mixing eras and keeps comparisons fair and consistent.",
    tags: ["Core", "Data"],
    where: PRODUCT_LINKS,
  },
  {
    term: "Team (abbreviation)",
    meaning:
      "A three-letter team code (e.g., TOR, BOS) used in dropdowns and requests.",
    whyItMatters:
      "Keeps the UI compact while still mapping to friendly team names when available.",
    tags: ["Core", "Data"],
    where: PRODUCT_LINKS,
    aliases: ["TOR", "BOS", "LAL"],
  },
  {
    term: "Our Team / Opponent",
    meaning:
      "The matchup inputs: our team is the offense being planned; opponent is the defense being targeted (or filtered against).",
    whyItMatters:
      "Most outputs are matchup-dependent: the best play types or shots can change versus different opponents.",
    tags: ["Core", "Context"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
      { label: "Shot Plan", href: "/shot-plan" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
    aliases: ["our", "opp"],
  },
  {
    term: "Top-K",
    meaning:
      "How many top recommendations the page returns (example: Top-5 play types).",
    whyItMatters:
      "Keeps the output coach-friendly: a short ranked list is easier to act on than a long table.",
    tags: ["Core", "Models"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
    ],
    aliases: ["k"],
  },
  // Dataset1 metrics + fields
  {
    term: "Play Type",
    meaning:
      "A category of offensive action (example: pick-and-roll, transition, spot-up). The product ranks play types, not individual set plays.",
    whyItMatters:
      "This is the main recommendation unit for Dataset1: what kinds of actions to emphasize versus an opponent.",
    tags: ["Core", "Dataset1"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
      { label: "Gameplan", href: "/gameplan" },
    ],
  },
  {
    term: "Side (offense / defense)",
    meaning:
      "Offense = points your team scores on a play type. Defense = points allowed by a team when defending that play type.",
    whyItMatters:
      "The matchup logic blends our offense with their defense-allowed to estimate what will work in a specific matchup.",
    tags: ["Data", "Dataset1"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
    ],
  },
  {
    term: "Possession (POSS)",
    meaning:
      "A single chance for a team to score. Dataset1 counts possessions used for each play type.",
    whyItMatters:
      "More possessions usually means more reliable estimates; low samples can look unrealistically good or bad.",
    tags: ["Data", "Dataset1"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
    ],
  },
  {
    term: "Usage % (POSS_PCT)",
    meaning:
      "How often a play type was used relative to a team’s total possessions.",
    whyItMatters:
      "Helps judge realism: a rarely-used play type may be efficient but not a practical primary option.",
    tags: ["Data", "Dataset1"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
    ],
    aliases: ["POSS%", "Usage"],
  },
  {
    term: "PPP (Points Per Possession)",
    meaning:
      "Average points scored per possession for a play type. Higher = more efficient offense (or worse defense if shown on defense side).",
    whyItMatters:
      "PPP is the primary efficiency metric used to rank play types.",
    tags: ["Metrics", "Dataset1", "Core"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
    ],
  },
  {
    term: "Shrinkage",
    meaning:
      "A statistical technique that pulls extreme values toward the league average when there’s limited data (low possessions).",
    whyItMatters:
      "Makes the baseline model more defensible by reducing overconfidence in tiny samples.",
    tags: ["Stats", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
  },
  {
    term: "Reliability Weight",
    meaning:
      "A 0–1 weight computed from possessions that controls how much we trust a team’s play-type stats vs the league average.",
    whyItMatters:
      "High possessions → high reliability → less shrinkage; low possessions → more shrinkage.",
    tags: ["Stats", "Dataset1"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "Data Explorer", href: "/data-explorer" },
    ],
    aliases: ["RELIABILITY_WEIGHT"],
  },
  {
    term: "Baseline Recommender",
    meaning:
      "An explainable model that blends our shrunk offense PPP with the opponent’s shrunk defense-allowed PPP to estimate a predicted PPP per play type.",
    whyItMatters:
      "It’s the transparent reference model used to validate and contextualize the AI page.",
    tags: ["Models", "Dataset1", "Core"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["Baseline ranking"],
  },
  {
    term: "Predicted PPP (Baseline)",
    meaning:
      "The baseline model’s output estimate for a play type in the selected matchup (a weighted blend).",
    whyItMatters:
      "This is the sortable value used to rank Top-K play types on the baseline page.",
    tags: ["Models", "Metrics", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["pppPred", "PPP_PRED"],
  },
  {
    term: "Gap (PPP Gap)",
    meaning:
      "A difference-style field used to communicate separation between offense strength and opponent defense-allowed context for that play type.",
    whyItMatters:
      "Helps explain why a play type rises or falls in a matchup beyond just the final rank.",
    tags: ["Metrics", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["pppGap"],
  },
  {
    term: "Weights (w_off / w_def)",
    meaning:
      "The user-tunable balance between our offense and the opponent’s defense-allowed when computing predicted PPP.",
    whyItMatters:
      "Lets you emphasize what you trust more (your offense or their defense) while keeping the formula explainable.",
    tags: ["Models", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["w_off", "w_def"],
  },
  {
    term: "Auto-normalized weights",
    meaning:
      "A safety step that forces weights to sum to 1 (even if the user enters values that don’t).",
    whyItMatters:
      "Keeps interpretation stable and prevents accidental ‘double counting’ in the formula.",
    tags: ["Engineering", "Models"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
  },
  {
    term: "Rationale",
    meaning:
      "A plain-English explanation field returned alongside rankings describing why a play type scored well.",
    whyItMatters:
      "Improves trust: reviewers can see the ‘reason’ in words, not just numbers.",
    tags: ["Core", "Explainability"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "AI Context", href: "/context" },
      { label: "Gameplan", href: "/gameplan" },
    ],
  },
  // Context / AI fields (Dataset1)
  {
    term: "PPP_BASELINE",
    meaning:
      "The baseline predicted PPP for a play type (what the baseline page shows).",
    whyItMatters:
      "It’s the reference point used to compute how much context-aware AI changes the recommendation.",
    tags: ["AI", "Context", "Dataset1"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["baselinePPP"],
  },
  {
    term: "PPP_ML_BLEND",
    meaning:
      "A model-based estimate for offensive efficiency used by the context workflow before applying context adjustments.",
    whyItMatters:
      "Represents the ‘AI’ part: it replaces purely historical offense PPP with a predictive signal.",
    tags: ["AI", "Models", "Dataset1"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["mlPPP"],
  },
  {
    term: "PPP_CONTEXT (Final PPP)",
    meaning:
      "The context-aware final efficiency value after applying situation logic (score/time) on top of ML/baseline signals.",
    whyItMatters:
      "This is what ultimately determines the context ranking order on the AI page.",
    tags: ["AI", "Context", "Metrics", "Dataset1"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["finalPPP"],
  },
  {
    term: "DELTA_VS_BASELINE",
    meaning:
      "How much the context-aware PPP differs from baseline PPP for the same play type.",
    whyItMatters:
      "Makes AI impact measurable: reviewers can see exactly how recommendations shift.",
    tags: ["AI", "Context", "Evaluation", "Dataset1"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["deltaPPP"],
  },
  {
    term: "Margin",
    meaning:
      "Our score minus opponent score. Negative = trailing, positive = leading.",
    whyItMatters:
      "Driving signal for late-game strategy changes (chasing points vs protecting a lead).",
    tags: ["Context", "Core"],
    where: [
      { label: "AI Context", href: "/context" },
      { label: "Gameplan", href: "/gameplan" },
    ],
  },
  {
    term: "Period / Time remaining",
    meaning:
      "Which quarter (or OT) and how many seconds remain in the current period.",
    whyItMatters:
      "Controls urgency (late-game factor) so context changes matter more when time is running out.",
    tags: ["Context", "Core"],
    where: [
      { label: "AI Context", href: "/context" },
      { label: "Gameplan", href: "/gameplan" },
    ],
  },
  {
    term: "Late factor",
    meaning:
      "A 0–1-ish urgency weight derived from period and time remaining (ramps up late in games).",
    whyItMatters:
      "Prevents overreacting early in games while still allowing meaningful late-game behavior.",
    tags: ["Context", "Stats"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["lateFactor"],
  },
  {
    term: "Policy overlay",
    meaning:
      "A small, explainable adjustment layer that nudges ranking based on intuitive coaching heuristics (late + trailing, late + leading, quick vs safe).",
    whyItMatters:
      "Keeps the AI screen understandable: you can describe how the ranking changed without black-box reasoning.",
    tags: ["AI", "Explainability", "Context"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["overlay"],
  },
  {
    term: "Variance proxy",
    meaning:
      "A simple ‘risk’ signal: how much ML and baseline disagree for a play type (bigger disagreement → treated as riskier).",
    whyItMatters:
      "Supports sensible late-game behavior: protect a lead by slightly penalizing high-variance options.",
    tags: ["Context", "AI", "Stats"],
    where: [{ label: "AI Context", href: "/context" }],
    aliases: ["varianceProxy"],
  },
  // Modeling + evaluation (Dataset1)
  {
    term: "Holdout evaluation (season splits)",
    meaning:
      "Testing where entire seasons are held out for evaluation instead of mixing training and testing within the same season.",
    whyItMatters:
      "More realistic: the model is evaluated on ‘future-like’ data it didn’t train on.",
    tags: ["Evaluation", "Models", "Dataset1"],
    where: [{ label: "Model Performance", href: "/model-metrics" }],
    aliases: ["n_splits", "season holdout"],
  },
  {
    term: "RMSE",
    meaning:
      "Root Mean Squared Error: average prediction error with larger errors penalized more (lower is better).",
    whyItMatters:
      "Primary ‘how wrong are we?’ metric used to compare regression models.",
    tags: ["Evaluation", "Metrics"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Shot Model Metrics", href: "/shot-model-metrics" },
    ],
  },
  {
    term: "MAE",
    meaning:
      "Mean Absolute Error: average absolute prediction error (lower is better).",
    whyItMatters:
      "A more interpretable error metric (not as sensitive to outliers as RMSE).",
    tags: ["Evaluation", "Metrics"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Shot Model Metrics", href: "/shot-model-metrics" },
    ],
  },
  {
    term: "R² (R2)",
    meaning:
      "Explained variance score: how much of the target variation is explained by the model (higher is better).",
    whyItMatters:
      "A complementary metric that captures how well the model tracks overall variation, not just average error.",
    tags: ["Evaluation", "Metrics"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Shot Model Metrics", href: "/shot-model-metrics" },
    ],
    aliases: ["R2_mean", "R2_std"],
  },
  {
    term: "Paired t-test (optional)",
    meaning:
      "A statistical test comparing two models’ errors across the same splits, returning a t-statistic and p-value.",
    whyItMatters:
      "Supports a stronger claim that one model’s improvement is not just random split noise (when provided).",
    tags: ["Evaluation", "Stats"],
    where: [{ label: "Model Performance", href: "/model-metrics" }],
    aliases: ["p-value", "rf_vs_baseline_p"],
  },
  {
    term: "Cross-validation (CV)",
    meaning:
      "A training workflow that evaluates many model/settings combinations across multiple folds to reduce overfitting during selection.",
    whyItMatters:
      "Provides evidence you searched systematically for good settings rather than picking arbitrary parameters.",
    tags: ["Evaluation", "Models", "AI"],
    where: [
      { label: "Statistical Analysis", href: "/statistical-analysis" },
      { label: "Model Performance", href: "/model-metrics" },
    ],
  },
  {
    term: "Hyperparameter tuning",
    meaning:
      "Trying different model settings (like depth/trees/regularization) and selecting the best based on CV performance.",
    whyItMatters:
      "Prevents the ‘we picked a model once and hoped’ critique; it shows a deliberate search process.",
    tags: ["Evaluation", "AI"],
    where: [
      { label: "Statistical Analysis", href: "/statistical-analysis" },
      { label: "Model Performance", href: "/model-metrics" },
    ],
  },
  {
    term: "Feature selection",
    meaning:
      "Choosing which input columns (features) are included in the ML model using analysis steps like correlation filtering or selection algorithms.",
    whyItMatters:
      "Improves model quality and keeps the approach defensible (you can explain what signals were used).",
    tags: ["Evaluation", "AI"],
    where: [
      { label: "Statistical Analysis", href: "/statistical-analysis" },
      { label: "Model Performance", href: "/model-metrics" },
    ],
  },
  {
    term: "Correlation filter",
    meaning:
      "A selection step that removes redundant features that are too highly correlated with others beyond a threshold.",
    whyItMatters:
      "Reduces multicollinearity and keeps models simpler and more stable.",
    tags: ["Evaluation", "Stats"],
    where: [{ label: "Statistical Analysis", href: "/statistical-analysis" }],
    aliases: ["threshold"],
  },
  {
    term: "SelectKBest",
    meaning:
      "A feature selection method that keeps the top K features according to a scoring rule.",
    whyItMatters:
      "Produces a defendable, compact feature set without manual guessing.",
    tags: ["Evaluation", "AI"],
    where: [{ label: "Statistical Analysis", href: "/statistical-analysis" }],
    aliases: ["k best"],
  },
  {
    term: "RFE (Recursive Feature Elimination)",
    meaning:
      "A method that repeatedly fits a model and removes the least useful features until a target set remains.",
    whyItMatters:
      "Another defendable way to narrow to a strong feature set without ad-hoc decisions.",
    tags: ["Evaluation", "AI"],
    where: [{ label: "Statistical Analysis", href: "/statistical-analysis" }],
  },
  {
    term: "Ridge regression",
    meaning:
      "A linear regression model with regularization (penalizes large coefficients).",
    whyItMatters:
      "Serves as a stable baseline ML model in the evaluation comparisons.",
    tags: ["Models", "AI"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Statistical Analysis", href: "/statistical-analysis" },
    ],
  },
  {
    term: "Random Forest",
    meaning:
      "An ensemble of decision trees that averages predictions to reduce overfitting and capture non-linear patterns.",
    whyItMatters:
      "Often performs well on structured tabular data; included as a strong candidate model.",
    tags: ["Models", "AI"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Statistical Analysis", href: "/statistical-analysis" },
    ],
  },
  {
    term: "Gradient Boosting",
    meaning:
      "An ensemble method that builds trees sequentially, each one correcting errors from the previous.",
    whyItMatters:
      "A common high-performing tabular model, used as another candidate in tuning and comparisons.",
    tags: ["Models", "AI"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Statistical Analysis", href: "/statistical-analysis" },
    ],
  },
  // Visuals + exports
  {
    term: "SportyPy court map",
    meaning:
      "A generated court visualization for a selected play type (returned as a PNG image).",
    whyItMatters:
      "Adds a real visual layer to recommendations so results feel coach-ready, not just numbers.",
    tags: ["Viz", "Exports", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
    aliases: ["Court map", "Playtype viz"],
  },
  {
    term: "Base64 image",
    meaning:
      "A way to send an image as text (a base64-encoded PNG) so the UI can render it instantly without extra file hosting.",
    whyItMatters:
      "Simplifies deployment: the backend returns a self-contained visualization payload.",
    tags: ["Engineering", "Viz"],
    aliases: ["image_base64"],
  },
  {
    term: "CSV export",
    meaning:
      "A downloadable spreadsheet-like file that matches the filters you used in the UI.",
    whyItMatters:
      "Supports external verification: Excel/R/Python checks using the exact same subset the UI showed.",
    tags: ["Exports", "Core"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "Shots Explorer", href: "/shot-explorer" },
      { label: "Shot Model Metrics", href: "/shot-model-metrics" },
    ],
  },
  {
    term: "1-page PDF export",
    meaning:
      "A single-page ‘coach handout’ style export for a selected play type visualization.",
    whyItMatters:
      "Makes the product feel complete and presentation-ready for real stakeholders.",
    tags: ["Exports", "Viz", "Dataset1"],
    where: [{ label: "Matchup (Baseline)", href: "/matchup" }],
  },
  // Dataset2 (shots) terminology
  {
    term: "Play-by-play (PBP)",
    meaning:
      "Event-level game data. In this product, it’s used specifically for shot events (one row per shot).",
    whyItMatters:
      "Provides a large, high-granularity dataset for spatial visuals and shot selection recommendations.",
    tags: ["Dataset2", "Data", "Core"],
    where: [
      { label: "Shots Explorer", href: "/shot-explorer" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shot Plan", href: "/shot-plan" },
    ],
  },
  {
    term: "Shot event (one row per shot)",
    meaning:
      "A single recorded shot attempt with attributes like team, opponent, location/zone, shot type, and outcome (as available).",
    whyItMatters:
      "It’s the raw foundation for shot aggregation, heatmaps, and shot-plan rankings.",
    tags: ["Dataset2", "Data"],
    where: [{ label: "Shots Explorer", href: "/shot-explorer" }],
    aliases: ["shot row"],
  },
  {
    term: "Shot Type",
    meaning:
      "A categorical label for the kind of attempt (example: layup, dunk, three-pointer type, etc., depending on dataset taxonomy).",
    whyItMatters:
      "Shot plans can recommend what types of attempts to prioritize for higher expected value.",
    tags: ["Dataset2", "Data"],
    where: [
      { label: "Shot Plan", href: "/shot-plan" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
  },
  {
    term: "Zone",
    meaning:
      "A categorical region of the court used for aggregation (example: rim/paint/midrange/corner three, depending on dataset taxonomy).",
    whyItMatters:
      "Enables spatial analysis and lets shot plans say ‘where’ to hunt shots, not just what type.",
    tags: ["Dataset2", "Data", "Viz"],
    where: [
      { label: "Shot Plan", href: "/shot-plan" },
      { label: "Shot Heatmap", href: "/shot-heatmap" },
      { label: "Shots Explorer", href: "/shot-explorer" },
    ],
  },
  {
    term: "Expected Points (shot value estimate)",
    meaning:
      "A model-based estimate of how many points a shot attempt is ‘worth on average’ given its characteristics.",
    whyItMatters:
      "Shot plans can rank options by expected value, not just raw frequency.",
    tags: ["Dataset2", "Models", "Metrics"],
    where: [{ label: "Shot Plan", href: "/shot-plan" }],
    aliases: ["Expected value", "EPA (if shown)"],
  },
  {
    term: "Max shots",
    meaning:
      "A cap controlling how many shot events to sample/use when generating a heatmap for speed and stability.",
    whyItMatters:
      "Large datasets can overwhelm rendering; capping keeps the UI responsive while still showing real patterns.",
    tags: ["Dataset2", "Engineering", "Viz"],
    where: [{ label: "Shot Heatmap", href: "/shot-heatmap" }],
    aliases: ["max_shots"],
  },
  {
    term: "Downsampling",
    meaning:
      "A method that uses a subset of rows rather than every row, usually for performance.",
    whyItMatters:
      "Lets the product handle huge shot datasets without freezing or long waits during demos.",
    tags: ["Dataset2", "Engineering"],
    where: [{ label: "Shot Heatmap", href: "/shot-heatmap" }],
  },
  // Dataset2 evaluation / leakage control
  {
    term: "GroupKFold by GAME_ID",
    meaning:
      "A cross-validation method that groups shots by game so shots from the same game are not split across train and test.",
    whyItMatters:
      "Avoids ‘game leakage’ where the model learns game-specific patterns and appears better than it truly generalizes.",
    tags: ["Dataset2", "Evaluation", "Engineering"],
    where: [{ label: "Shot Model Metrics", href: "/shot-model-metrics" }],
    aliases: ["GAME_ID grouping", "No leakage"],
  },
  {
    term: "Data leakage",
    meaning:
      "When information from the test set accidentally influences training, inflating reported performance.",
    whyItMatters:
      "Leakage can make a model look unrealistically good; preventing it is key for credible evaluation.",
    tags: ["Evaluation", "Engineering"],
    where: [
      { label: "Model Performance", href: "/model-metrics" },
      { label: "Shot Model Metrics", href: "/shot-model-metrics" },
    ],
  },
  // Gameplan NLP / interpretation (as implemented)
  {
    term: "Natural language interpretation (NLP → structured context)",
    meaning:
      "Turning a coaching-style sentence into structured fields the product understands (time/score intent + flags).",
    whyItMatters:
      "Makes the product feel coach-friendly while keeping the downstream logic defendable (everything becomes explicit fields).",
    tags: ["AI", "Core", "Explainability"],
    where: [{ label: "Gameplan", href: "/gameplan" }],
    aliases: ["NLP parse", "parse"],
  },
  {
    term: "Confidence (NLP)",
    meaning:
      "A score indicating how confident the parser is about the structured context it extracted from the user’s text.",
    whyItMatters:
      "Supports a defendable UX: if confidence is low, the UI can prompt for clarification rather than guessing.",
    tags: ["AI", "Explainability"],
    where: [{ label: "Gameplan", href: "/gameplan" }],
    aliases: ["confidence score"],
  },
  {
    term: "Clarifying question",
    meaning:
      "A follow-up prompt suggested when the user’s input is ambiguous (example: missing time or whether they need a 2 or 3).",
    whyItMatters:
      "Prevents ‘AI hallucination’ behavior: the system asks instead of inventing assumptions.",
    tags: ["AI", "Explainability"],
    where: [{ label: "Gameplan", href: "/gameplan" }],
  },
  // A few “common UI labels” that appear across screens
  {
    term: "Explainable / Trust anchor",
    meaning:
      "A design principle used in this app: baseline pages show transparent, repeatable logic before any advanced AI behavior.",
    whyItMatters:
      "Makes the capstone defensible: stakeholders can validate the baseline and then assess what AI changes (and why).",
    tags: ["Core", "Explainability"],
    where: [
      { label: "Matchup (Baseline)", href: "/matchup" },
      { label: "Model Performance", href: "/model-metrics" },
    ],
  },
  {
    term: "Reproducibility",
    meaning:
      "The ability for someone else to rerun the same filters/inputs and get the same outputs.",
    whyItMatters:
      "A key committee requirement for credible analytics: exports + clear inputs make results verifiable.",
    tags: ["Core", "Evaluation"],
    where: [
      { label: "Data Explorer", href: "/data-explorer" },
      { label: "Shots Explorer", href: "/shot-explorer" },
      { label: "Matchup (Baseline)", href: "/matchup" },
    ],
  },
];

function pillStyle(active: boolean): CSSProperties {
  return {
    borderRadius: 999,
    padding: "6px 10px",
    border: "1px solid rgba(15, 23, 42, 0.10)",
    background: active
      ? "linear-gradient(135deg, rgba(56,189,248,0.22), rgba(99,102,241,0.18), rgba(34,197,94,0.12))"
      : "rgba(255,255,255,0.60)",
    fontSize: 12,
    cursor: "pointer",
    userSelect: "none",
    boxShadow: active ? "0 8px 22px rgba(15, 23, 42, 0.10)" : "none",
  };
}

export default function GlossaryPage() {
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState<string>("All");

  const allTags = useMemo(() => {
    const set = new Set<string>();
    TERMS.forEach((t) => (t.tags ?? []).forEach((x) => set.add(x)));
    return ["All", ...Array.from(set).sort((a, b) => a.localeCompare(b))];
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return TERMS.filter((t) => {
      const matchesTag = tag === "All" ? true : (t.tags ?? []).includes(tag);
      if (!matchesTag) return false;

      if (!q) return true;

      const whereBlob = (t.where ?? []).map((w) => `${w.label} ${w.href}`).join(" ");
      const aliasBlob = (t.aliases ?? []).join(" ");
      const blob = `${t.term} ${t.meaning} ${t.whyItMatters ?? ""} ${whereBlob} ${aliasBlob}`.toLowerCase();
      return blob.includes(q);
    });
  }, [query, tag]);

  const counts = useMemo(() => {
    const withWhy = filtered.filter((t) => Boolean(t.whyItMatters)).length;
    const withWhere = filtered.filter((t) => (t.where?.length ?? 0) > 0).length;
    return { total: filtered.length, withWhy, withWhere };
  }, [filtered]);

  const heroStyle: CSSProperties = {
    borderRadius: 18,
    padding: "18px 18px 14px",
    background:
      "linear-gradient(135deg, rgba(56,189,248,0.16), rgba(99,102,241,0.14), rgba(34,197,94,0.10))",
    border: "1px solid rgba(255,255,255,0.10)",
  };

  return (
    <main className="page" style={{ paddingBottom: 56 }}>
      {/* HERO */}
      <header className="page__header" style={heroStyle}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <h1 className="h1" style={{ margin: 0 }}>
              Glossary
            </h1>
            <p className="muted" style={{ marginTop: 6, fontSize: 14, marginBottom: 0 }}>
              Plain-English definitions for terms used across the entire product built for fast review and easy traceability.
            </p>
          </div>
        </div>

        {/* ✅ Keep only: Search bar + counters + tag pills (no Product map) */}
        <div
          style={{
            marginTop: 14,
            display: "flex",
            gap: 10,
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              Search
              <input
                className="input"
                style={{ width: 320, maxWidth: "100%" }}
                placeholder="Try: PPP, shrinkage, GroupKFold, shot type, Top-K, baseline…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>

            <span className="badge">
              Showing {counts.total} {counts.total === 1 ? "term" : "terms"}
            </span>
            <span className="badge blue">With “Why it matters”: {counts.withWhy}</span>
            <span className="badge">With “Where you see it”: {counts.withWhere}</span>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {allTags.map((t) => (
              <span
                key={t}
                role="button"
                tabIndex={0}
                style={pillStyle(tag === t)}
                onClick={() => setTag(t)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setTag(t);
                }}
                aria-label={`Filter glossary by tag ${t}`}
                title={`Filter: ${t}`}
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </header>

      {/* CONTENT */}
      <section className="card" style={{ marginTop: 14 }}>
        {filtered.length === 0 ? (
          <div style={{ padding: 8 }}>
            <p className="muted" style={{ margin: 0 }}>
              No terms match your search. Try removing filters or searching a shorter keyword.
            </p>
          </div>
        ) : (
          <div style={{ marginTop: 4, display: "grid", gap: 12 }}>
            {filtered.map((t) => (
              <article
                key={t.term}
                style={{
                  border: "1px solid rgba(15, 23, 42, 0.10)",
                  borderRadius: 16,
                  padding: 14,
                  background: "linear-gradient(135deg, rgba(255,255,255,0.70), rgba(255,255,255,0.55))",
                  boxShadow: "0 10px 26px rgba(15, 23, 42, 0.06)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 10,
                    alignItems: "baseline",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
                    <h2 style={{ margin: 0, fontSize: 16 }}>{t.term}</h2>
                    {t.tags?.length ? (
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {t.tags.map((x) => (
                          <span key={x} className="badge">
                            {x}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>

                  {t.whyItMatters ? <span className="badge blue">Why it matters</span> : null}
                </div>

                <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
                  {t.meaning}
                </p>

                {t.where?.length ? (
                  <div style={{ marginTop: 10 }}>
                    <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                      Where you see it:
                    </div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {t.where.map((w) => (
                        <Link
                          key={`${t.term}-${w.href}-${w.label}`}
                          href={w.href}
                          className="badge"
                          style={{
                            textDecoration: "none",
                            background: "rgba(255,255,255,0.55)",
                            borderColor: "rgba(15,23,42,0.12)",
                          }}
                        >
                          {w.label}
                        </Link>
                      ))}
                    </div>
                  </div>
                ) : null}

                {t.whyItMatters ? (
                  <div
                    style={{
                      marginTop: 10,
                      borderRadius: 14,
                      padding: 12,
                      border: "1px solid rgba(29, 66, 138, 0.18)",
                      background: "rgba(29, 66, 138, 0.06)",
                    }}
                  >
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <span
                        className="badge"
                        style={{
                          background: "rgba(29,66,138,0.12)",
                          borderColor: "rgba(29,66,138,0.20)",
                        }}
                      >
                        Context
                      </span>
                      <span style={{ fontSize: 12, color: "rgba(15,23,42,0.85)" }}>{t.whyItMatters}</span>
                    </div>
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        )}

        <div
          style={{
            marginTop: 16,
            display: "flex",
            gap: 10,
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <Link className="btn" href="/model-metrics">
            Back: Model Performance
          </Link>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button
              className="btn btn--secondary"
              type="button"
              onClick={() => {
                setQuery("");
                setTag("All");
                try {
                  window.scrollTo({ top: 0, behavior: "smooth" });
                } catch {
                  window.scrollTo(0, 0);
                }
              }}
            >
              Clear filters
            </button>
            <Link className="btn" href="/">
              Home
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}