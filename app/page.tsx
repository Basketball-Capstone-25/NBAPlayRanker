export default function Page() {
  return (
    <div className="grid">
      {/* Hero / overview card */}
      <section className="card">
        <h1 className="h1">Basketball Game Strategy Analysis</h1>
        <p className="muted">
          A small web slice that lets coaches and analysts explore Synergy play-type data
          and a simple baseline recommendation model for a single game matchup.
        </p>
        <div className="kpi">
          <span className="pill">FastAPI + Next.js</span>
          <span className="pill">Synergy play-types 2019–25</span>
          <span className="pill">Baseline play recommendations</span>
        </div>
      </section>

      {/* Tiles: short descriptions for each main page */}
      <section className="card">
        <h2>What you can do on each page</h2>
        <ul className="muted" style={{ fontSize: 14, paddingLeft: 20 }}>
          <li>
            <strong>Home</strong> – High-level overview of the product slice and how the
            other pages fit together.
          </li>
          <li>
            <strong>Data Explorer</strong> – Filter and preview the underlying
            team–season–play-type data that feeds the baseline model and export it as CSV
            for deeper offline analysis.
          </li>
          <li>
            <strong>Matchup Console</strong> – See a worked example of the baseline
            Top-K recommendations for a single matchup. Used mainly to explain the logic.
          </li>
          <li>
            <strong>Context Simulator</strong> – Adjust score, quarter, and time remaining
            to show where a future context-aware ML model would live. Currently still
            backed by the same baseline logic.
          </li>
          <li>
            <strong>Model Performance</strong> – Compare the baseline model against more
            advanced ML models (e.g., Ridge, Random Forest) on historical data using
            RMSE/MAE/R².
          </li>
          <li>
            <strong>Glossary</strong> – Quick definitions of key basketball and modelling
            terms used across the app.
          </li>
        </ul>
      </section>

      {/* Short architecture / notes card */}
      <section className="card">
        <h2>How this slice fits the capstone</h2>
        <p className="muted">
          The frontend is a Next.js App Router UI. It calls a FastAPI backend that loads a
          Synergy CSV snapshot, prepares team-level offense/defense tables, and runs a
          baseline ranking formula for a given matchup. This PSPI focuses on proving the
          architecture and interaction flow; later releases can layer on richer models and
          visual polish.
        </p>
      </section>
    </div>
  );
}
