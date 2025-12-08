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
      def: "How many points per possession the opponent usually gives up on that play type in the Synergy snapshot.",
    },
    {
      term: "Predicted PPP (baseline)",
      def: "The baseline model’s estimate of PPP for a play type in a given matchup, combining our offense and the opponent’s defense.",
    },
    {
      term: "Gap vs allowed",
      def: "Predicted PPP minus the opponent’s usual allowed PPP on that play type. Positive = better than what they normally give up.",
    },
    {
      term: "Top-K recommendations",
      def: "The K highest-ranked play types (e.g., Top 5 or Top 10) returned by the baseline model for a matchup.",
    },
    {
      term: "RMSE / MAE / R²",
      def: "Standard regression metrics used on the Model Performance page to compare the baseline model to ML models.",
    },
  ];

  return (
    <section className="card">
      <h1 className="h1">Glossary</h1>
      <p className="muted">
        Key terms used throughout the Basketball Game Strategy Analysis app.
      </p>
      <ul style={{ marginTop: 12 }}>
        {items.map((i) => (
          <li key={i.term} style={{ marginBottom: 8 }}>
            <b>{i.term}:</b> {i.def}
          </li>
        ))}
      </ul>
    </section>
  );
}
