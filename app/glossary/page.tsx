export default function Glossary() {
  const items = [
    {
      term: "Play type",
      def: "How an offensive possession is labelled in the Synergy data (e.g., Pick-and-Roll Ball-Handler, Spot-Up, Isolation).",
    },
    {
      term: "PPP (Points Per Possession)",
      def: "Total points scored divided by the number of possessions. Higher PPP means a more efficient play type.",
    },
    {
      term: "Opponent allowed PPP",
      def: "How many points per possession the opponent usually gives up on that play type (from the Synergy snapshot).",
    },
    {
      term: "Predicted PPP (baseline)",
      def: "A simple blend of our offensive PPP and the opponent’s defensive PPP allowed. It is a lightweight heuristic model, not a final ML model.",
    },
    {
      term: "Top-K ranking",
      def: "Ordering play types from best to worst and showing only the top K results (for example, the best 5 or 10 options).",
    },
    {
      term: "Context-aware model (future work)",
      def: "A richer ML model that would take score margin, clock, period, and other game context into account when ranking plays.",
    },
  ];

  return (
    <section className="card">
      <h1 className="h1">Glossary</h1>
      <p className="muted">
        Quick terms you can reference while demoing the product.
      </p>
      <ul>
        {items.map((i) => (
          <li key={i.term} style={{ marginBottom: 8 }}>
            <b>{i.term}:</b> {i.def}
          </li>
        ))}
      </ul>
    </section>
  );
}
