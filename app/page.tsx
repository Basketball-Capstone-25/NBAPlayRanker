import Link from "next/link";
import styles from "./home.module.css";

const workflows = [
  { number: "01", title: "Find the matchup", text: "Compare your offense with the opponent’s defense.", href: "/matchup", action: "Open Baseline" },
  { number: "02", title: "Read the numbers", text: "Explore play usage, efficiency and season trends.", href: "/matchup-dashboard", action: "Open matchup dashboard" },
  { number: "03", title: "Plan the next possession", text: "Add the score and time, then review the options.", href: "/gameplan", action: "Build a gameplan" },
];

export default function Page() {
  return <section className={styles.page}>
    <section className={styles.hero}>
      <div className={styles.copy}>
        <p className={styles.eyebrow}>NBA PLAY RANKER</p>
        <h1>Know the matchup.<br /><span>Make your next move.</span></h1>
        <p className={styles.lead}>Compare play types, spot defensive openings and build a gameplan with the numbers behind every choice.</p>
        <div className={styles.actions}>
          <Link className={styles.primary} href="/login">Open your workspace <span aria-hidden>→</span></Link>
          <Link className={styles.secondary} href="/signup">Create an account</Link>
        </div>
        <p className={styles.note}>Historical data. Clear comparisons. Your coaching judgment.</p>
      </div>
      <div className={styles.court} aria-hidden="true">
        <div className={styles.courtLabel}>THE NEXT POSSESSION</div>
        <svg viewBox="0 0 360 330" fill="none">
          <g stroke="currentColor" strokeWidth="1.5">
            <path d="M25 25H335V305H25Z M120 25V160H240V25 M25 75C25 335 335 335 335 75 M25 305H335" />
            <circle cx="180" cy="160" r="45" /><path d="M150 58H210 M180 58V68" /><circle cx="180" cy="78" r="10" />
          </g>
          <path d="M73 230Q137 207 180 102" stroke="#b4a5ff" strokeWidth="3" strokeDasharray="7 7" />
          <path d="M172 108L180 98L184 111" stroke="#b4a5ff" strokeWidth="3" />
          <circle cx="73" cy="230" r="17" fill="#b4a5ff" /><text x="73" y="236" textAnchor="middle" fill="#19203c" fontSize="17" fontWeight="700">1</text>
          <circle cx="287" cy="187" r="17" fill="#20334f" stroke="#b4a5ff" strokeWidth="2" /><text x="287" y="193" textAnchor="middle" fill="#ddd6ff" fontSize="17" fontWeight="700">2</text>
        </svg>
        <div className={styles.courtCaption}><span>Compare</span><span>Understand</span><span>Plan</span></div>
      </div>
    </section>

    <section className={styles.workflow} aria-labelledby="workflow-heading">
      <div className={styles.sectionHeading}><h2 id="workflow-heading">From matchup to gameplan</h2><p>A clear next step, every time.</p></div>
      <div className={styles.steps}>{workflows.map(item => <Link key={item.number} href={item.href} className={styles.step}>
        <span className={styles.number}>{item.number}</span><h3>{item.title}</h3><p>{item.text}</p><span className={styles.stepAction}>{item.action} <span aria-hidden>→</span></span>
      </Link>)}</div>
    </section>

    <section className={styles.analyst} aria-labelledby="analyst-heading">
      <div><p className={styles.kicker}>ANALYST WORKSPACE</p><h2 id="analyst-heading">Go straight to the evidence.</h2><p>Browse the source data, inspect shot locations and check model performance.</p></div>
      <div className={styles.tools}><Link href="/data-explorer">Data Explorer <span aria-hidden>→</span></Link><Link href="/model-metrics">Model Metrics <span aria-hidden>→</span></Link><Link href="/shot-heatmap">Shot Heatmap <span aria-hidden>→</span></Link></div>
    </section>
    <p className={styles.access}>Workspace access requires an approved coach or analyst account. <Link href="/glossary">New to the terms?</Link></p>
  </section>;
}
