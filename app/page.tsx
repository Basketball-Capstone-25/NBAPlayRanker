// app/page.tsx
//
// Homepage
// - premium basketball-first landing
// - fully responsive
// - no horizontal scrolling
// - no extra packages required
// - Server Component

import Link from "next/link";

const sellingPoints = [
  {
    title: "Rank play types with a clear baseline",
    desc: "Start from a transparent ranking of offensive play types and courses of action using historical matchup data before layering in live game pressure.",
  },
  {
    title: "Adapt recommendations to game context",
    desc: "See live recommendations update for score, clock, and possession context so the ranked options stay useful as the game changes.",
  },
  {
    title: "Support coaching and player collaboration",
    desc: "Turn ranked outputs into clearer on-court collaboration with visuals, supporting evidence, and a gameplan that is easier to communicate to players.",
  },
];

const proofCards = [
  {
    title: "Matchup / Baseline",
    desc: "A transparent starting point for every opponent. See the initial ranking before context shifts the decision.",
    href: "/matchup",
    tag: "Coach",
  },
  {
    title: "Context / ML",
    desc: "Re-rank recommendations for score, time, and pressure so late-game decisions reflect the situation on the floor.",
    href: "/context",
    tag: "Coach",
  },
  {
    title: "Gameplan",
    desc: "A cleaner end-state for decision making: what to prioritize, what to prepare, and what to communicate.",
    href: "/gameplan",
    tag: "Coach",
  },
  {
    title: "Data Explorer",
    desc: "Inspect the supporting data directly and move from recommendation to underlying evidence without losing context.",
    href: "/data-explorer",
    tag: "Analyst",
  },
  {
    title: "Model Metrics",
    desc: "Review performance evidence so the recommendation is supported by measured validation, not just output.",
    href: "/model-metrics",
    tag: "Analyst",
  },
  {
    title: "Statistical Analysis",
    desc: "Extend the workflow with deeper quantitative review for analyst interpretation and decision support.",
    href: "/statistical-analysis",
    tag: "Analyst",
  },
];

const audienceCards = [
  {
    title: "For coaches",
    bullets: [
      "See priority play types faster before film room discussions get crowded.",
      "Adjust recommendations for score, time, and pressure without losing the trust anchor.",
      "Walk into pre-game and late-game planning with a cleaner, more presentable decision flow.",
    ],
  },
  {
    title: "For analysts",
    bullets: [
      "Move from surfaced recommendation to underlying data, metrics, and statistical review.",
      "Compare baseline and context-aware behavior in a way that stays readable and defensible.",
      "Support coaching conversations with evidence that is structured for explanation, not just inspection.",
    ],
  },
];

export default function Page() {
  return (
    <main className="hp-page">
      <style suppressHydrationWarning>{`
        :root {
          --hp-bg: #f7f9fc;
          --hp-line: rgba(15,23,42,0.09);
          --hp-line-strong: rgba(15,23,42,0.14);
          --hp-text: #0f172a;
          --hp-muted: rgba(15,23,42,0.68);
          --hp-soft: rgba(15,23,42,0.48);
          --hp-blue: #2563eb;
          --hp-purple: #7c3aed;
          --hp-pink: #ec4899;
          --hp-shadow: 0 28px 80px rgba(15,23,42,0.12);
        }

        html, body {
          max-width: 100%;
          overflow-x: hidden;
        }

        * {
          box-sizing: border-box;
        }

        .hp-page {
          width: 100%;
          max-width: 100%;
          overflow-x: hidden;
          padding: 18px 0 28px;
        }

        .hp-wrap {
          width: min(1450px, calc(100% - 28px));
          margin: 0 auto;
        }

        .hp-shell {
          position: relative;
          width: 100%;
          max-width: 100%;
          overflow: hidden;
          border-radius: 30px;
          border: 1px solid var(--hp-line);
          background:
            radial-gradient(1100px 560px at 8% -6%, rgba(37,99,235,0.18), transparent 58%),
            radial-gradient(1200px 620px at 102% 2%, rgba(124,58,237,0.15), transparent 56%),
            radial-gradient(980px 520px at 65% 115%, rgba(236,72,153,0.12), transparent 58%),
            linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.72)),
            var(--hp-bg);
          box-shadow: var(--hp-shadow);
        }

        .hp-shell::before {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.18), transparent 22%),
            radial-gradient(700px 120px at 50% 0%, rgba(255,255,255,0.35), transparent 70%);
          opacity: 0.9;
        }

        .hp-hero {
          position: relative;
          z-index: 1;
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 18px;
          padding: 18px;
          align-items: stretch;
        }

        @media (min-width: 1120px) {
          .hp-hero {
            grid-template-columns: minmax(0, 1.02fr) minmax(0, 0.98fr);
            gap: 22px;
          }
        }

        .hp-copy,
        .hp-preview,
        .hp-surface,
        .hp-surface-body,
        .hp-preview-grid,
        .hp-mini-grid,
        .hp-why-grid,
        .hp-workflow-grid,
        .hp-proof-grid,
        .hp-audience-grid {
          min-width: 0;
          max-width: 100%;
        }

        .hp-copy {
          display: grid;
          align-content: center;
          gap: 16px;
        }

        .hp-kicker {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          width: fit-content;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid var(--hp-line);
          background: rgba(255,255,255,0.7);
          color: rgba(15,23,42,0.74);
          font-size: 12px;
          font-weight: 950;
          letter-spacing: 0.02em;
          box-shadow: 0 12px 26px rgba(15,23,42,0.04);
        }

        .hp-kicker::before {
          content: "";
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
          box-shadow: 0 0 0 4px rgba(124,58,237,0.08);
        }

        .hp-title {
          margin: 0;
          font-size: clamp(38px, 6vw, 78px);
          line-height: 0.94;
          letter-spacing: -0.06em;
          font-weight: 1000;
          color: var(--hp-text);
          max-width: 11ch;
        }

        .hp-title .hp-gradient {
          background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 42%, #7c3aed 72%, #ec4899 100%);
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
        }

        .hp-sub {
          margin: 0;
          max-width: 68ch;
          font-size: 16px;
          line-height: 1.75;
          color: var(--hp-muted);
        }

        .hp-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-items: center;
          margin-top: 4px;
        }

        .hp-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 52px;
          padding: 0 20px;
          border-radius: 18px;
          text-decoration: none;
          font-size: 14px;
          font-weight: 950;
          letter-spacing: -0.02em;
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, filter 160ms ease, background 160ms ease;
        }

        .hp-btn:hover {
          transform: translateY(-1px);
        }

        .hp-btn-primary {
          color: #fff;
          border: 1px solid rgba(255,255,255,0.22);
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
          box-shadow: 0 18px 40px rgba(124,58,237,0.22);
        }

        .hp-btn-primary:hover {
          filter: brightness(1.04);
          box-shadow: 0 24px 48px rgba(124,58,237,0.28);
        }

        .hp-btn-secondary {
          color: rgba(15,23,42,0.88);
          border: 1px solid var(--hp-line-strong);
          background: rgba(255,255,255,0.76);
          box-shadow: 0 14px 30px rgba(15,23,42,0.06);
        }

        .hp-btn-secondary:hover {
          background: rgba(255,255,255,0.92);
          border-color: rgba(37,99,235,0.2);
        }

        .hp-proofline {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: center;
          color: var(--hp-soft);
          font-size: 13px;
          font-weight: 800;
        }

        .hp-proofline span {
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }

        .hp-proofline span::before {
          content: "";
          width: 6px;
          height: 6px;
          border-radius: 999px;
          background: rgba(37,99,235,0.4);
        }

        .hp-mini-grid {
          display: grid;
          gap: 12px;
          margin-top: 4px;
        }

        @media (min-width: 760px) {
          .hp-mini-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }

        .hp-mini-card {
          border-radius: 18px;
          border: 1px solid var(--hp-line);
          background: rgba(255,255,255,0.7);
          padding: 14px 14px 15px;
          box-shadow: 0 14px 32px rgba(15,23,42,0.05);
          backdrop-filter: blur(8px);
        }

        .hp-mini-card h3 {
          margin: 0;
          font-size: 13px;
          font-weight: 950;
          letter-spacing: -0.02em;
          color: rgba(15,23,42,0.92);
        }

        .hp-mini-card p {
          margin: 8px 0 0;
          font-size: 13px;
          line-height: 1.6;
          color: var(--hp-muted);
        }

        .hp-preview {
          border-radius: 26px;
          border: 1px solid var(--hp-line);
          background:
            radial-gradient(700px 260px at 24% 0%, rgba(37,99,235,0.12), transparent 58%),
            radial-gradient(780px 260px at 100% 8%, rgba(124,58,237,0.12), transparent 58%),
            radial-gradient(620px 280px at 60% 100%, rgba(236,72,153,0.09), transparent 58%),
            rgba(255,255,255,0.7);
          box-shadow: 0 28px 60px rgba(15,23,42,0.08);
          overflow: hidden;
          min-width: 0;
        }

        .hp-surface {
          margin: 14px;
          border-radius: 24px;
          border: 1px solid rgba(15,23,42,0.08);
          background:
            linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.5)),
            rgba(248,250,252,0.74);
          overflow: hidden;
        }

        .hp-surface-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          padding: 14px 16px;
          border-bottom: 1px solid rgba(15,23,42,0.07);
          background: rgba(255,255,255,0.64);
        }

        .hp-tabs {
          display: inline-flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .hp-tab {
          padding: 7px 11px;
          border-radius: 999px;
          border: 1px solid rgba(15,23,42,0.08);
          background: rgba(255,255,255,0.7);
          color: rgba(15,23,42,0.72);
          font-size: 11px;
          font-weight: 900;
          letter-spacing: 0.01em;
          white-space: nowrap;
        }

        .hp-tab.active {
          color: #fff;
          border-color: rgba(255,255,255,0.18);
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
          box-shadow: 0 12px 28px rgba(124,58,237,0.16);
        }

        .hp-tag {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(15,23,42,0.08);
          background: rgba(15,23,42,0.03);
          color: rgba(15,23,42,0.68);
          font-size: 11px;
          font-weight: 900;
          white-space: nowrap;
        }

        .hp-tag::before {
          content: "";
          width: 7px;
          height: 7px;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
        }

        .hp-surface-body {
          padding: 16px;
          display: grid;
          gap: 14px;
        }

        .hp-reco-card,
        .hp-context-card,
        .hp-heat-card,
        .hp-metric-card,
        .hp-note-card {
          border-radius: 20px;
          border: 1px solid rgba(15,23,42,0.08);
          background: rgba(255,255,255,0.84);
          box-shadow: 0 18px 34px rgba(15,23,42,0.06);
          min-width: 0;
        }

        .hp-reco-card {
          padding: 14px;
        }

        .hp-reco-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 10px;
          margin-bottom: 12px;
        }

        .hp-command-kicker {
          font-size: 11px;
          font-weight: 950;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: rgba(37,99,235,0.84);
        }

        .hp-command-title {
          margin: 4px 0 0;
          font-size: 20px;
          line-height: 1.05;
          letter-spacing: -0.04em;
          font-weight: 1000;
          color: var(--hp-text);
        }

        .hp-score-chip {
          display: inline-flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 4px;
          padding: 8px 10px;
          border-radius: 14px;
          border: 1px solid rgba(15,23,42,0.08);
          background: rgba(15,23,42,0.03);
          min-width: 96px;
        }

        .hp-score-chip strong {
          font-size: 16px;
          font-weight: 1000;
          letter-spacing: -0.03em;
          color: var(--hp-text);
        }

        .hp-score-chip span {
          font-size: 11px;
          font-weight: 900;
          color: var(--hp-soft);
        }

        .hp-ranked {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
        }

        .hp-rank {
          border-radius: 14px;
          border: 1px solid rgba(15,23,42,0.08);
          background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.64));
          padding: 10px;
          min-width: 0;
        }

        .hp-rank-num {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          border-radius: 999px;
          background: rgba(37,99,235,0.08);
          color: rgba(15,23,42,0.88);
          font-size: 11px;
          font-weight: 1000;
          margin-bottom: 8px;
        }

        .hp-rank:nth-child(2) .hp-rank-num {
          background: rgba(124,58,237,0.08);
        }

        .hp-rank:nth-child(3) .hp-rank-num {
          background: rgba(236,72,153,0.08);
        }

        .hp-rank-title {
          margin: 0;
          font-size: 12px;
          font-weight: 950;
          color: rgba(15,23,42,0.92);
        }

        .hp-rank-sub {
          margin: 5px 0 0;
          font-size: 11.5px;
          line-height: 1.5;
          color: var(--hp-soft);
        }

        .hp-preview-grid {
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 14px;
        }

        @media (min-width: 700px) {
          .hp-preview-grid {
            grid-template-columns: minmax(0, 1.25fr) minmax(220px, 0.75fr);
          }
        }

        .hp-context-card,
        .hp-metric-card,
        .hp-note-card {
          padding: 12px;
        }

        .hp-context-card h4,
        .hp-heat-card h4,
        .hp-metric-card h4,
        .hp-note-card h4 {
          margin: 0;
          font-size: 12px;
          font-weight: 950;
          color: rgba(15,23,42,0.92);
        }

        .hp-context-lines,
        .hp-metric-row {
          margin-top: 10px;
          display: grid;
          gap: 8px;
        }

        .hp-context-line,
        .hp-metric {
          border-radius: 12px;
          border: 1px solid rgba(15,23,42,0.07);
          background: rgba(15,23,42,0.03);
          padding: 8px;
        }

        .hp-context-line strong {
          display: block;
          font-size: 10px;
          font-weight: 950;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--hp-soft);
        }

        .hp-context-line span {
          display: block;
          margin-top: 4px;
          font-size: 12px;
          font-weight: 900;
          color: var(--hp-text);
        }

        .hp-metric strong {
          display: block;
          font-size: 14px;
          line-height: 1;
          font-weight: 1000;
          color: var(--hp-text);
        }

        .hp-metric span {
          display: block;
          margin-top: 5px;
          font-size: 10.5px;
          font-weight: 900;
          color: var(--hp-soft);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .hp-note-card p {
          margin: 8px 0 0;
          font-size: 11.5px;
          line-height: 1.6;
          color: var(--hp-muted);
        }

        .hp-heat-card {
          padding: 14px;
        }

        .hp-heat-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
          flex-wrap: wrap;
        }

        .hp-court {
          position: relative;
          height: 220px;
          border-radius: 18px;
          border: 1px solid rgba(15,23,42,0.08);
          background:
            linear-gradient(180deg, rgba(15,23,42,0.025), rgba(15,23,42,0.01)),
            repeating-linear-gradient(
              90deg,
              rgba(15,23,42,0.03) 0px,
              rgba(15,23,42,0.03) 1px,
              transparent 1px,
              transparent 22px
            ),
            radial-gradient(circle at 26% 72%, rgba(37,99,235,0.18), transparent 14%),
            radial-gradient(circle at 52% 42%, rgba(124,58,237,0.2), transparent 16%),
            radial-gradient(circle at 74% 24%, rgba(236,72,153,0.18), transparent 14%),
            rgba(255,255,255,0.92);
          overflow: hidden;
        }

        .hp-court::before {
          content: "";
          position: absolute;
          inset: 0;
          background:
            radial-gradient(circle at 50% 50%, transparent 0 34px, rgba(15,23,42,0.16) 35px 36px, transparent 37px),
            linear-gradient(90deg, transparent 0 49.65%, rgba(15,23,42,0.16) 49.65% 50.35%, transparent 50.35% 100%);
          opacity: 0.55;
        }

        .hp-court::after {
          content: "";
          position: absolute;
          inset: 12px;
          border-radius: 14px;
          border: 1px dashed rgba(15,23,42,0.16);
          opacity: 0.6;
        }

        .hp-zone {
          position: absolute;
          width: 16px;
          height: 16px;
          border-radius: 999px;
          z-index: 1;
        }

        .hp-zone::before,
        .hp-zone::after {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: 999px;
        }

        .hp-zone::before {
          transform: scale(2.2);
          opacity: 0.18;
          background: currentColor;
        }

        .hp-zone::after {
          transform: scale(4.1);
          opacity: 0.08;
          background: currentColor;
        }

        .hp-zone.z1 { top: 68%; left: 24%; background: rgba(37,99,235,0.96); color: rgba(37,99,235,1); }
        .hp-zone.z2 { top: 44%; left: 50%; background: rgba(124,58,237,0.96); color: rgba(124,58,237,1); }
        .hp-zone.z3 { top: 20%; left: 74%; background: rgba(236,72,153,0.94); color: rgba(236,72,153,1); }
        .hp-zone.z4 { top: 58%; left: 66%; background: rgba(59,130,246,0.9); color: rgba(59,130,246,1); }

        .hp-marquee {
          position: relative;
          z-index: 1;
          width: 100%;
          max-width: 100%;
          overflow: hidden;
          border-top: 1px solid var(--hp-line);
          border-bottom: 1px solid var(--hp-line);
          background: rgba(255,255,255,0.42);
        }

        .hp-marquee-track {
          display: flex;
          width: max-content;
          animation: hpMarquee 26s linear infinite;
          will-change: transform;
        }

        .hp-marquee-item {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 14px 18px;
          color: rgba(15,23,42,0.72);
          font-size: 13px;
          font-weight: 900;
          letter-spacing: -0.01em;
          white-space: nowrap;
        }

        .hp-marquee-item::before {
          content: "";
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
          box-shadow: 0 0 0 5px rgba(37,99,235,0.08);
        }

        .hp-section {
          position: relative;
          z-index: 1;
          padding: 20px 18px;
        }

        .hp-section-head {
          display: flex;
          justify-content: space-between;
          align-items: end;
          gap: 14px;
          flex-wrap: wrap;
          margin-bottom: 14px;
        }

        .hp-eyebrow {
          margin: 0 0 6px;
          font-size: 12px;
          font-weight: 950;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(37,99,235,0.84);
        }

        .hp-section-title {
          margin: 0;
          font-size: clamp(24px, 3.2vw, 40px);
          line-height: 1.02;
          letter-spacing: -0.045em;
          font-weight: 980;
          color: var(--hp-text);
        }

        .hp-section-sub {
          margin: 0;
          max-width: 66ch;
          font-size: 14px;
          line-height: 1.7;
          color: var(--hp-muted);
        }

        .hp-proof-grid {
          display: grid;
          gap: 12px;
        }

        @media (min-width: 980px) {
          .hp-proof-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }

        .hp-proof-card {
          border-radius: 22px;
          border: 1px solid var(--hp-line);
          background: rgba(255,255,255,0.75);
          padding: 16px;
          text-decoration: none;
          box-shadow: 0 18px 36px rgba(15,23,42,0.05);
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
        }

        .hp-proof-card:hover {
          transform: translateY(-2px);
          border-color: rgba(37,99,235,0.18);
          background: rgba(255,255,255,0.92);
          box-shadow: 0 24px 46px rgba(15,23,42,0.08);
        }

        .hp-proof-top {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          align-items: center;
          margin-bottom: 10px;
        }

        .hp-proof-tag {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 26px;
          padding: 0 10px;
          border-radius: 999px;
          border: 1px solid var(--hp-line);
          background: rgba(15,23,42,0.03);
          color: rgba(15,23,42,0.68);
          font-size: 11px;
          font-weight: 950;
          letter-spacing: 0.05em;
        }

        .hp-proof-arrow {
          color: rgba(15,23,42,0.38);
          font-size: 20px;
          line-height: 1;
          font-weight: 700;
        }

        .hp-proof-card h3 {
          margin: 0;
          font-size: 17px;
          line-height: 1.1;
          letter-spacing: -0.03em;
          font-weight: 980;
          color: var(--hp-text);
        }

        .hp-proof-card p {
          margin: 10px 0 0;
          font-size: 14px;
          line-height: 1.7;
          color: var(--hp-muted);
        }

        .hp-audience-grid {
          display: grid;
          gap: 12px;
        }

        @media (min-width: 900px) {
          .hp-audience-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        .hp-audience-card {
          border-radius: 24px;
          border: 1px solid var(--hp-line);
          background: rgba(255,255,255,0.74);
          padding: 18px;
          box-shadow: 0 18px 36px rgba(15,23,42,0.05);
        }

        .hp-audience-card h3 {
          margin: 0;
          font-size: 22px;
          line-height: 1;
          font-weight: 980;
          letter-spacing: -0.04em;
          color: var(--hp-text);
        }

        .hp-audience-card ul {
          margin: 14px 0 0;
          padding: 0;
          list-style: none;
          display: grid;
          gap: 10px;
        }

        .hp-audience-card li {
          display: grid;
          grid-template-columns: 18px 1fr;
          gap: 10px;
          align-items: start;
          font-size: 14px;
          line-height: 1.7;
          color: var(--hp-muted);
        }

        .hp-audience-card li::before {
          content: "";
          width: 10px;
          height: 10px;
          border-radius: 999px;
          margin-top: 6px;
          background: linear-gradient(135deg, var(--hp-blue) 0%, var(--hp-purple) 55%, var(--hp-pink) 100%);
          box-shadow: 0 0 0 5px rgba(37,99,235,0.08);
        }

        @keyframes hpMarquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }

        @media (max-width: 880px) {
          .hp-ranked {
            grid-template-columns: 1fr;
          }

          .hp-preview-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 720px) {
          .hp-page {
            padding-top: 14px;
          }

          .hp-wrap {
            width: calc(100% - 18px);
          }

          .hp-shell {
            border-radius: 24px;
          }

          .hp-hero,
          .hp-section {
            padding-left: 14px;
            padding-right: 14px;
          }

          .hp-title {
            max-width: 10.5ch;
          }

          .hp-sub {
            font-size: 15px;
          }

          .hp-proofline {
            flex-direction: column;
            align-items: flex-start;
          }

          .hp-surface {
            margin: 10px;
            border-radius: 20px;
          }

          .hp-court {
            height: 180px;
          }
        }
      `}</style>

      <div className="hp-wrap">
        <section className="hp-shell">
          <section className="hp-hero">
            <div className="hp-copy">
              <div className="hp-kicker">Historical data, live recommendations, and context-aware decision support</div>

              <h1 className="hp-title">
                Rank the right <span className="hp-gradient">play type</span> for the next decision.
              </h1>

              <p className="hp-sub">
                NBA Play Ranker helps coaches and analysts rank offensive play types and courses of action using
                historical data, then adapt those recommendations live as score, time, and game context change. The
                platform is built to support clearer player collaboration, faster coaching communication, and decisions
                that stay explainable from scouting report to gameplan.
              </p>

              <div className="hp-actions">
                <Link href="/signup" className="hp-btn hp-btn-primary">
                  Create account
                </Link>
                <Link href="/login" className="hp-btn hp-btn-secondary">
                  Log in to explore
                </Link>
              </div>

              <div className="hp-proofline">
                <span>Ranks play types and courses of action</span>
                <span>Live recommendations adapted to game context</span>
                <span>Historical data and evidence for player collaboration</span>
              </div>

              <div className="hp-mini-grid">
                {sellingPoints.map((item) => (
                  <div key={item.title} className="hp-mini-card">
                    <h3>{item.title}</h3>
                    <p>{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <aside className="hp-preview" aria-label="NBA Play Ranker product preview">
              <div className="hp-surface">
                <div className="hp-surface-top">
                  <div className="hp-tabs">
                    <span className="hp-tab">Matchup</span>
                    <span className="hp-tab">Explorer</span>
                    <span className="hp-tab active">Gameplan</span>
                    <span className="hp-tab">Evidence</span>
                  </div>

                  <div className="hp-tag">Coach-facing flow</div>
                </div>

                <div className="hp-surface-body">
                  <div className="hp-reco-card">
                    <div className="hp-reco-head">
                      <div>
                        <div className="hp-command-kicker">Opponent setup</div>
                        <p className="hp-command-title">TOR vs BOS</p>
                      </div>

                      <div className="hp-score-chip">
                        <strong>1:12</strong>
                        <span>Late-game context</span>
                      </div>
                    </div>

                    <div className="hp-ranked">
                      <div className="hp-rank">
                        <div className="hp-rank-num">1</div>
                        <p className="hp-rank-title">PnR Ball Handler</p>
                        <p className="hp-rank-sub">Strong baseline profile with stable value under pressure.</p>
                      </div>

                      <div className="hp-rank">
                        <div className="hp-rank-num">2</div>
                        <p className="hp-rank-title">Handoff</p>
                        <p className="hp-rank-sub">Useful entry option against the matchup shell and help timing.</p>
                      </div>

                      <div className="hp-rank">
                        <div className="hp-rank-num">3</div>
                        <p className="hp-rank-title">Spot Up</p>
                        <p className="hp-rank-sub">Reliable secondary action when coverage collapses late.</p>
                      </div>
                    </div>
                  </div>

                  <div className="hp-preview-grid">
                    <div className="hp-heat-card">
                      <div className="hp-heat-head">
                        <h4>Court view</h4>
                        <div className="hp-tag">Visual proof</div>
                      </div>

                      <div className="hp-court">
                        <span className="hp-zone z1" />
                        <span className="hp-zone z2" />
                        <span className="hp-zone z3" />
                        <span className="hp-zone z4" />
                      </div>
                    </div>

                    <div style={{ display: "grid", gap: "14px", minWidth: 0 }}>
                      <div className="hp-context-card">
                        <h4>Scenario</h4>
                        <div className="hp-context-lines">
                          <div className="hp-context-line">
                            <strong>Score</strong>
                            <span>+2</span>
                          </div>
                          <div className="hp-context-line">
                            <strong>Time</strong>
                            <span>1:12 left</span>
                          </div>
                          <div className="hp-context-line">
                            <strong>Pressure</strong>
                            <span>Possession-heavy</span>
                          </div>
                        </div>
                      </div>

                      <div className="hp-metric-card">
                        <h4>Evidence</h4>
                        <div className="hp-metric-row">
                          <div className="hp-metric">
                            <strong>Holdout</strong>
                            <span>Model check</span>
                          </div>
                          <div className="hp-metric">
                            <strong>Compare</strong>
                            <span>Baseline vs context</span>
                          </div>
                          <div className="hp-metric">
                            <strong>Review</strong>
                            <span>Stats + explorer</span>
                          </div>
                        </div>
                      </div>

                      <div className="hp-note-card">
                        <h4>Decision use</h4>
                        <p>Built for pre-game planning, scenario prep, and clearer coaching conversations.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </aside>
          </section>

          <div className="hp-marquee" aria-hidden="true">
            <div className="hp-marquee-track">
              {[
                "Transparent matchup ranking",
                "Late-game scenario planning",
                "Court visuals",
                "Explainable recommendation flow",
                "Model evidence",
                "Statistical review",
                "Analyst traceability",
                "Coach-ready output",
                "Transparent matchup ranking",
                "Late-game scenario planning",
                "Court visuals",
                "Explainable recommendation flow",
                "Model evidence",
                "Statistical review",
                "Analyst traceability",
                "Coach-ready output",
              ].map((item, index) => (
                <span key={`${item}-${index}`} className="hp-marquee-item">
                  {item}
                </span>
              ))}
            </div>
          </div>

          <section className="hp-section" id="proof">
            <div className="hp-section-head">
              <div>
                <p className="hp-eyebrow">Platform surfaces</p>
                <h2 className="hp-section-title">Every key surface supports the same decision flow.</h2>
              </div>

              <p className="hp-section-sub">
                Matchup, context, gameplan, explorer, and evidence pages are designed to work together so users can
                move between recommendation and validation without losing continuity.
              </p>
            </div>

            <div className="hp-proof-grid">
              {proofCards.map((card) => (
                <Link key={card.title} href={card.href} className="hp-proof-card">
                  <div className="hp-proof-top">
                    <span className="hp-proof-tag">{card.tag}</span>
                    <span className="hp-proof-arrow">→</span>
                  </div>
                  <h3>{card.title}</h3>
                  <p>{card.desc}</p>
                </Link>
              ))}
            </div>
          </section>

          <section className="hp-section">
            <div className="hp-section-head">
              <div>
                <p className="hp-eyebrow">Roles</p>
                <h2 className="hp-section-title">Built for the people who have to use the answer.</h2>
              </div>

              <p className="hp-section-sub">
                Coaches need a faster path to action. Analysts need a stronger path to verification. The product is
                designed to support both without making either workflow feel secondary.
              </p>
            </div>

            <div className="hp-audience-grid">
              {audienceCards.map((card) => (
                <div key={card.title} className="hp-audience-card">
                  <h3>{card.title}</h3>
                  <ul>
                    {card.bullets.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}