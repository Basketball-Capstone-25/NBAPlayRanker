export default function Page() {
  return (
    <div className="grid">
      <section className="card">
        <h1 className="h1">Basketball Game Strategy Analysis</h1>
        <p className="muted">Capstone PSPI – baseline play-type recommender demo.</p>
        <div className="kpi">
          <span className="pill">FastAPI + Next.js</span>
          <span className="pill">Synergy play-types 2019–25</span>
          <span className="pill">Top-K play recommendations</span>
        </div>
      </section>

      <section className="card">
        <h2>How to demo in under a minute</h2>
        <ol>
          <li>Open <b>Matchup Console</b>, pick a season, your team, and an opponent.</li>
          <li>Explain that the UI calls the FastAPI <code>/rank-plays/baseline</code> endpoint.</li>
          <li>Show the ranked Top-K list and point out the highlighted top recommendation.</li>
          <li>Switch to <b>Data Explorer</b> to browse the same results as a simple table.</li>
        </ol>
      </section>

      <section className="card">
        <h2>Screens in this PSPI</h2>
        <ul>
          <li><b>Data Explorer</b> – browse baseline rankings (Top 10 plays) for any matchup.</li>
          <li><b>Matchup Console</b> – coach view: Top-K list with a clear best play call.</li>
          <li><b>Context Simulator</b> – UI mock that sets up the future ML, using baseline results for now.</li>
          <li><b>Glossary</b> – quick definitions for PPP, play types, and Top-K terms.</li>
        </ul>
      </section>

      <section className="card">
        <h2>Notes</h2>
        <p className="muted">
          This is a UI-focused product slice. The backend loads a Synergy-based CSV snapshot,
          runs a simple baseline formula, and exposes it via FastAPI. The goal of this PSPI is
          to prove the architecture and interaction flow, not final design polish.
        </p>
      </section>
    </div>
  );
}
