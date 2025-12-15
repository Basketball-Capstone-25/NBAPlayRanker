// app/page.tsx
//
// Home page (non-basketball + non-technical).
//
// Goals:
// - Explain the real-world problem in plain English.
// - Make it obvious why there are TWO recommenders:
//   (1) Baseline = transparent math
//   (2) Context+ML = AI + game context adjustments
// - Provide a dead-simple “demo flow” so reviewers know what to click.
// - Summarize architecture at a high level (frontend -> API -> data/model).
//
// NOTE: This is intentionally static (server component). No hooks needed.

import Link from "next/link";

export default function Page() {
  return (
    <div className="grid">
      {/* HERO / PURPOSE */}
      <section className="card">
        <h1 className="h1">Basketball Strategy Decision Support</h1>
        <p className="muted">
          This capstone is a <strong>decision-support</strong> web app. It helps a coach
          or analyst quickly answer:
          <br />
          <strong>“Which offensive actions should we prioritize against this opponent?”</strong>
        </p>

        <div className="kpi">
          <span className="pill">Explainable baseline</span>
          <span className="pill">AI + context adjustments</span>
          <span className="pill">Exportable dataset</span>
        </div>

        <p className="muted" style={{ marginTop: 12 }}>
          You do <strong>not</strong> need basketball knowledge to evaluate this project.
          Think of “play types” as categories of actions (like “pick-and-roll”, “spot-up”).
          The app ranks which actions are most likely to be efficient against a specific
          opponent.
        </p>
      </section>

      {/* WHAT PROBLEM / WHY IT MATTERS */}
      <section className="card">
        <h2>What real problem does this solve?</h2>
        <p className="muted">
          Coaches and analysts often have a lot of statistics, but during games they need a
          fast, defensible way to narrow down options for a specific opponent.
          This tool turns historical play-type data into a ranked list that is:
        </p>
        <ul className="muted" style={{ fontSize: 14, paddingLeft: 20 }}>
          <li>
            <strong>Explainable</strong> (baseline model shows the math and “why”)
          </li>
          <li>
            <strong>Data-driven</strong> (all outputs trace back to the dataset)
          </li>
          <li>
            <strong>Context-aware</strong> (AI/ML + game state changes priorities)
          </li>
          <li>
            <strong>Reproducible</strong> (model metrics page shows evaluation results)
          </li>
        </ul>
      </section>

      {/* SIMPLE WORKFLOW (LINKS TO PAGES) */}
      <section className="card">
        <h2>Demo flow (click in this order)</h2>
        <ol className="muted" style={{ fontSize: 14, paddingLeft: 20 }}>
          <li>
            <Link href="/data-explorer">
              <strong>Data Explorer</strong>
            </Link>{" "}
            — Preview the cleaned team-level dataset used by the models, filter it, and export CSV.
          </li>
          <li>
            <Link href="/matchup">
              <strong>Matchup Console (Baseline)</strong>
            </Link>{" "}
            — Generates Top-K recommendations using an explainable formula and shows the breakdown.
          </li>
          <li>
            <Link href="/context">
              <strong>Context Simulator (AI)</strong>
            </Link>{" "}
            — Uses ML predictions + game context (score/time) to adjust the ranking and shows what changed.
          </li>
          <li>
            <Link href="/model-metrics">
              <strong>Model Performance</strong>
            </Link>{" "}
            — Shows seasonal holdout evaluation and compares baseline vs ML (defends model choice).
          </li>
          <li>
            <Link href="/glossary">
              <strong>Glossary</strong>
            </Link>{" "}
            — Plain-English definitions used throughout the UI.
          </li>
        </ol>

        <p className="muted" style={{ marginTop: 12 }}>
          <strong>What the committee should observe:</strong> Each recommendation page returns
          both the ranking and the reasoning fields (formula inputs, shrinkage weights, context
          adjustments). Nothing is “magic.”
        </p>
      </section>

      {/* WHY AI IS NEEDED */}
      <section className="card">
        <h2>Where AI is used (and why it’s needed)</h2>
        <p className="muted">
          The project intentionally separates two layers:
        </p>
        <ul className="muted" style={{ fontSize: 14, paddingLeft: 20 }}>
          <li>
            <strong>Baseline recommender</strong>: purely statistical + explainable. It blends
            our historical efficiency with what the opponent typically allows, with shrinkage to
            reduce small-sample noise.
          </li>
          <li>
            <strong>Context + ML recommender</strong>: replaces historical offense efficiency with
            an ML-predicted efficiency and then applies small, transparent adjustments for game
            context (late game, trailing/leading).
          </li>
        </ul>

        <p className="muted" style={{ marginTop: 12 }}>
          This makes it clear which use cases require AI: the <strong>context simulator</strong>{" "}
          is the AI use case, while the baseline page remains the explainable reference.
        </p>
      </section>

      {/* ARCHITECTURE (HIGH LEVEL) */}
      <section className="card">
        <h2>System architecture (high level)</h2>
        <p className="muted">
          The app is a standard multi-user web architecture: a Next.js UI calling a FastAPI backend.
          The backend loads the dataset once (cached in memory per server worker) and exposes endpoints
          for data preview/export, recommendations, and model evaluation metrics.
        </p>

        <pre
          style={{
            background: "rgba(0,0,0,0.05)",
            padding: 12,
            borderRadius: 12,
            overflowX: "auto",
            fontSize: 13,
          }}
        >
{`Next.js UI (pages)
  ├─ Data Explorer  → GET /data/team-playtypes (+ CSV export)
  ├─ Baseline       → GET /rank-plays/baseline (+ breakdown fields)
  ├─ Context + ML   → GET /rank-plays/context-ml (+ context deltas)
  └─ Metrics        → GET /metrics/baseline-vs-ml (season holdout)

FastAPI Backend
  ├─ Loads Synergy snapshot CSV (startup)
  ├─ Aggregates to team-level play-type table
  ├─ Baseline recommender (shrinkage + blend vs opponent)
  ├─ ML predictions (precomputed CSV) for AI page
  └─ Evaluation (RMSE/MAE/R²) for defense justification`}
        </pre>

        <p className="muted" style={{ marginTop: 12 }}>
          This architecture also supports multiple users because requests are stateless, and heavy data/model
          loading is cached at server startup (per worker).
        </p>
      </section>

      {/* ARTIFACTS / WHAT TO SUBMIT*/}
      <section className="card">
        <h2>Project artefacts (for review)</h2>
        <p className="muted">
          In your final submission package, ensure these are included/linked clearly:
        </p>
        <ul className="muted" style={{ fontSize: 14, paddingLeft: 20 }}>
          <li>GitHub repository (code)</li>
          <li>Jira board (iterations, backlog comments, links to VP/code commits)</li>
          <li>Visual Paradigm diagrams (requirements, sequence, domain model, architecture)</li>
          <li>Dataset description + pipeline notes (see Data Explorer / Model Performance pages)</li>
        </ul>
        <p className="muted" style={{ marginTop: 8 }}>
          Tip: Put these links on this page (or the footer) before you re-defend so reviewers can access them quickly.
        </p>
      </section>
    </div>
  );
}
