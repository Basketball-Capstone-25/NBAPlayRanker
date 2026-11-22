"use client";

import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Sparkles,
  ShieldCheck,
  Gauge,
  GitCompare,
  Eye,
  Workflow,
  CheckCircle2,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 },
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const previewRows = [
  { name: "PnR Ball Handler", note: "Strong vs switching", score: 0.86 },
  { name: "Spot Up", note: "High-value catch & shoot", score: 0.79 },
  { name: "Transition", note: "Punish misses / long rebounds", score: 0.74 },
  { name: "Horns", note: "Organized late-clock entry", score: 0.68 },
];

const contexts = ["After timeout", "Late clock", "Need a 3", "Protect lead", "End of Q"];

export default function HomeLandingClient() {
  return (
    <div className="page">
      <div className="bg">
        <div className="noise" />
        <div className="vignette" />
        <div className="grid" />
        <div className="scan" />
        {/* floating orbs */}
        <motion.div
          className="orb o1"
          animate={{ y: [0, -18, 0], x: [0, 12, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="orb o2"
          animate={{ y: [0, 14, 0], x: [0, -10, 0] }}
          transition={{ duration: 9.5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="orb o3"
          animate={{ y: [0, -10, 0], x: [0, -14, 0] }}
          transition={{ duration: 10.5, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <div className="wrap">
        {/* HERO */}
        <section className="hero">
          <motion.div
            className="heroLeft"
            variants={stagger}
            initial="hidden"
            animate="show"
          >
            <motion.div className="kicker" variants={fadeUp}>
              <span className="kDot" />
              Coach-facing Decision Support • Explainable + Evidence-first
            </motion.div>

            <motion.h1 className="h1" variants={fadeUp}>
              Turn scouting into a{" "}
              <span className="grad">clear, explainable</span>{" "}
              gameplan.
            </motion.h1>

            <motion.p className="sub" variants={fadeUp}>
              NBA PlayRanker recommends <strong>which offensive play types to prioritize</strong>{" "}
              against a specific opponent. You get a <strong>transparent baseline</strong>, a{" "}
              <strong>context simulator</strong> for late-game situations, and{" "}
              <strong>visual proof</strong> so coaches can trust it fast.
            </motion.p>

            <motion.div className="ctaRow" variants={fadeUp}>
              <Link className="btn primary" href="/gameplan">
                Open Gameplan <ArrowRight size={18} />
              </Link>

              <Link className="btn ghost" href="/model-metrics">
                Evidence & performance <ShieldCheck size={18} />
              </Link>

              <div className="miniNote">
                Built for planning + “if/then” scenarios. Not gambling. Not fantasy.
              </div>
            </motion.div>

            <motion.div className="trustRow" variants={fadeUp}>
              <div className="trustCard">
                <div className="tIcon">
                  <GitCompare size={16} />
                </div>
                <div className="tText">
                  <div className="tTitle">Baseline → Context deltas</div>
                  <div className="tSub">See what changes and why.</div>
                </div>
              </div>

              <div className="trustCard">
                <div className="tIcon">
                  <Gauge size={16} />
                </div>
                <div className="tText">
                  <div className="tTitle">Holdout validation</div>
                  <div className="tSub">Model performance shown in basketball terms.</div>
                </div>
              </div>

              <div className="trustCard">
                <div className="tIcon">
                  <Eye size={16} />
                </div>
                <div className="tText">
                  <div className="tTitle">Visual strategy board</div>
                  <div className="tSub">Court views that communicate quickly.</div>
                </div>
              </div>
            </motion.div>

            {/* subtle marquee */}
            <div className="marquee" aria-hidden="true">
              <div className="marqueeTrack">
                <span>Matchup</span><span>Baseline</span><span>Context</span><span>Gameplan</span><span>Court View</span><span>Traceability</span>
                <span>Matchup</span><span>Baseline</span><span>Context</span><span>Gameplan</span><span>Court View</span><span>Traceability</span>
              </div>
            </div>
          </motion.div>

          {/* RIGHT PREVIEW PANEL */}
          <motion.aside
            className="heroRight"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <div className="panelTop">
              <div>
                <div className="panelTitle">Live preview (example)</div>
                <div className="panelSub">Matchup → baseline → context re-rank → visuals</div>
              </div>
              <div className="badge">
                <Sparkles size={14} />
                PSPI-ready
              </div>
            </div>

            {/* optional real hero image (low opacity) */}
            <div className="panelMedia">
              <div className="panelMediaInner">
                <Image
                  src="/landing/court-hero.jpg"
                  alt="Basketball court background"
                  fill
                  sizes="(max-width: 980px) 92vw, 520px"
                  priority
                  className="panelImg"
                />
                <div className="panelMediaOverlay" />
                <PlaybookCourt />
              </div>
            </div>

            <div className="panelRow">
              <div className="pillRow">
                <span className="pill">2024–25</span>
                <span className="pill">TOR</span>
                <span className="pill muted">vs</span>
                <span className="pill">BOS</span>
                <span className="pill muted">Top-K: 5</span>
              </div>

              <div className="ctxRow">
                {contexts.map((c) => (
                  <span key={c} className="ctx">
                    {c}
                  </span>
                ))}
              </div>
            </div>

            <div className="panelGrid">
              <div className="miniCard">
                <div className="miniHead">
                  <div className="miniTitle">Top recommendations</div>
                  <div className="miniTag">Baseline + context</div>
                </div>

                <div className="rows">
                  {previewRows.map((r) => (
                    <div key={r.name} className="row">
                      <div className="rowMain">
                        <div className="rowName">{r.name}</div>
                        <div className="rowNote">{r.note}</div>
                      </div>
                      <div className="rowBar">
                        <span style={{ width: `${Math.round(r.score * 100)}%` }} />
                      </div>
                      <div className="rowScore">{Math.round(r.score * 100)}%</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="miniCard">
                <div className="miniHead">
                  <div className="miniTitle">Coach workflow</div>
                  <div className="miniTag">60 seconds</div>
                </div>

                <div className="steps">
                  <div className="s">
                    <span className="sDot" />
                    <div>
                      <div className="sT">Pick matchup</div>
                      <div className="sD">Baseline trust anchor.</div>
                    </div>
                  </div>
                  <div className="s">
                    <span className="sDot" />
                    <div>
                      <div className="sT">Add context</div>
                      <div className="sD">Late-game “if/then” prep.</div>
                    </div>
                  </div>
                  <div className="s">
                    <span className="sDot" />
                    <div>
                      <div className="sT">Show visuals</div>
                      <div className="sD">Court view + counters.</div>
                    </div>
                  </div>
                </div>

                <div className="miniFooter">
                  <CheckCircle2 size={16} />
                  Exportable + reviewable evidence pages included.
                </div>
              </div>
            </div>
          </motion.aside>
        </section>

        {/* VALUE SECTION */}
        <section className="section">
          <motion.div
            className="sectionHead"
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 0.55 }}
          >
            <div className="eyebrow">What this feels like in practice</div>
            <h2 className="h2">
              A scouting assistant that speaks <span className="grad">basketball</span>.
            </h2>
            <p className="p">
              Less theory. More “what do we run next, and why?” — with proof you can show in a meeting.
            </p>
          </motion.div>

          <div className="cards">
            <FeatureCard
              icon={<GitCompare size={18} />}
              title="Baseline is the trust anchor"
              desc="Transparent ranking you can verify and repeat — before any ML context adjustments."
            />
            <FeatureCard
              icon={<Workflow size={18} />}
              title="Context simulator for late-game"
              desc="Score/time constraints re-rank your Top-K so you’re ready for the next timeout."
            />
            <FeatureCard
              icon={<Eye size={18} />}
              title="Court visuals for communication"
              desc="Playbook-style diagrams that explain spacing, intent, and counters quickly."
            />
            <FeatureCard
              icon={<ShieldCheck size={18} />}
              title="Evidence-first pages"
              desc="Holdout metrics, tuning traceability, and data explorer views for reviewers."
            />
          </div>
        </section>

        {/* BOTTOM CTA */}
        <section className="ctaBottom">
          <div className="ctaBox">
            <div>
              <div className="ctaTitle">Want the “wow” demo flow?</div>
              <div className="ctaSub">
                Start in Gameplan, then compare Baseline vs Context, and verify results in Model Performance.
              </div>
            </div>

            <div className="ctaBtns">
              <Link className="btn primary" href="/gameplan">
                Launch Gameplan <ArrowRight size={18} />
              </Link>
              <Link className="btn ghost" href="/matchup">
                See Baseline <GitCompare size={18} />
              </Link>
            </div>
          </div>
        </section>
      </div>

      <style jsx>{`
        .page {
          position: relative;
          min-height: 100vh;
          padding: 18px 0 80px;
          color: rgba(226, 232, 240, 0.92);
          overflow: hidden;
        }

        .bg {
          position: absolute;
          inset: 0;
          background:
            radial-gradient(1200px 700px at 18% -10%, rgba(59,130,246,0.28), transparent 55%),
            radial-gradient(900px 640px at 110% 10%, rgba(168,85,247,0.20), transparent 52%),
            radial-gradient(880px 600px at 54% 110%, rgba(236,72,153,0.16), transparent 55%),
            linear-gradient(180deg, rgba(2,6,23,0.98), rgba(2,6,23,0.92));
        }

        .noise {
          position: absolute;
          inset: 0;
          opacity: 0.08;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='.55'/%3E%3C/svg%3E");
          mix-blend-mode: overlay;
          pointer-events: none;
        }

        .vignette {
          position: absolute;
          inset: -2px;
          background: radial-gradient(circle at 50% 40%, transparent 0 55%, rgba(0,0,0,0.55) 78%, rgba(0,0,0,0.75) 100%);
          pointer-events: none;
        }

        .grid {
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px);
          background-size: 48px 48px;
          opacity: 0.14;
          mask-image: radial-gradient(circle at 50% 35%, rgba(0,0,0,1), rgba(0,0,0,0.15) 55%, transparent 80%);
          animation: drift 14s linear infinite;
          pointer-events: none;
        }

        .scan {
          position: absolute;
          left: 0;
          right: 0;
          top: -30%;
          height: 45%;
          background: linear-gradient(180deg, transparent, rgba(255,255,255,0.06), transparent);
          opacity: 0.18;
          transform: skewY(-10deg);
          animation: scan 6.5s ease-in-out infinite;
          pointer-events: none;
        }

        @keyframes drift {
          0% { background-position: 0 0, 0 0; }
          100% { background-position: 240px 240px, 240px 240px; }
        }
        @keyframes scan {
          0% { transform: translateY(-10%) skewY(-10deg); }
          50% { transform: translateY(160%) skewY(-10deg); }
          100% { transform: translateY(-10%) skewY(-10deg); }
        }

        .orb {
          position: absolute;
          border-radius: 999px;
          filter: blur(22px);
          opacity: 0.55;
          pointer-events: none;
        }
        .o1 { width: 240px; height: 240px; left: 6%; top: 18%; background: rgba(59,130,246,0.35); }
        .o2 { width: 220px; height: 220px; right: 9%; top: 14%; background: rgba(168,85,247,0.28); }
        .o3 { width: 260px; height: 260px; left: 44%; bottom: 10%; background: rgba(236,72,153,0.18); }

        .wrap {
          position: relative;
          max-width: 1180px;
          margin: 0 auto;
          padding: 0 16px;
        }

        .hero {
          display: grid;
          gap: 18px;
          align-items: stretch;
          margin-top: 10px;
        }
        @media (min-width: 980px) {
          .hero {
            grid-template-columns: 1.08fr 0.92fr;
            gap: 18px;
          }
        }

        .heroLeft, .heroRight {
          border-radius: 22px;
          border: 1px solid rgba(148,163,184,0.16);
          background: linear-gradient(180deg, rgba(15,23,42,0.62), rgba(15,23,42,0.44));
          box-shadow: 0 30px 80px rgba(0,0,0,0.35);
          overflow: hidden;
          position: relative;
        }
        .heroLeft { padding: 18px 18px 16px; }
        .heroRight { padding: 14px; }

        .heroLeft::before {
          content: "";
          position: absolute;
          inset: 0;
          background:
            radial-gradient(680px 360px at 12% 0%, rgba(59,130,246,0.16), transparent 55%),
            radial-gradient(680px 360px at 88% 12%, rgba(168,85,247,0.12), transparent 55%),
            radial-gradient(680px 360px at 50% 120%, rgba(236,72,153,0.10), transparent 55%);
          pointer-events: none;
        }

        .kicker {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 10px;
          font-weight: 800;
          font-size: 12px;
          color: rgba(226,232,240,0.78);
          border: 1px solid rgba(148,163,184,0.16);
          padding: 8px 12px;
          border-radius: 999px;
          background: rgba(2,6,23,0.35);
          backdrop-filter: blur(10px);
        }
        .kDot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          background: linear-gradient(135deg, rgba(59,130,246,1), rgba(168,85,247,1), rgba(236,72,153,1));
          box-shadow: 0 0 0 6px rgba(59,130,246,0.14);
        }

        .h1 {
          position: relative;
          margin: 14px 0 0;
          font-weight: 950;
          letter-spacing: -0.04em;
          line-height: 1.03;
          font-size: clamp(34px, 4.2vw, 56px);
          color: rgba(241,245,249,0.96);
        }
        .grad {
          background: linear-gradient(90deg, #60a5fa, #a78bfa, #fb7185);
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
        }

        .sub {
          position: relative;
          margin: 12px 0 0;
          font-size: clamp(14px, 1.35vw, 17px);
          line-height: 1.62;
          color: rgba(226,232,240,0.76);
          max-width: 58ch;
        }

        .ctaRow {
          position: relative;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 10px;
          margin-top: 16px;
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          border-radius: 14px;
          padding: 12px 14px;
          text-decoration: none;
          font-weight: 900;
          border: 1px solid rgba(148,163,184,0.16);
          transition: transform 160ms ease, filter 160ms ease, background 160ms ease;
          will-change: transform;
        }
        .btn:hover { transform: translateY(-1px); filter: brightness(1.06); }
        .btn:active { transform: translateY(0px) scale(0.99); }

        .primary {
          background: linear-gradient(135deg, rgba(59,130,246,1), rgba(168,85,247,0.95), rgba(236,72,153,0.95));
          color: rgba(2,6,23,0.96);
          border-color: rgba(255,255,255,0.12);
          box-shadow: 0 18px 40px rgba(59,130,246,0.18);
        }
        .ghost {
          background: rgba(2,6,23,0.30);
          color: rgba(241,245,249,0.90);
          backdrop-filter: blur(10px);
        }

        .miniNote {
          font-size: 12px;
          font-weight: 700;
          color: rgba(226,232,240,0.62);
          margin-left: 4px;
        }

        .trustRow {
          position: relative;
          display: grid;
          gap: 10px;
          margin-top: 16px;
        }
        @media (min-width: 720px) {
          .trustRow { grid-template-columns: repeat(3, 1fr); }
        }

        .trustCard {
          display: flex;
          gap: 10px;
          padding: 12px;
          border-radius: 16px;
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(2,6,23,0.22);
          backdrop-filter: blur(10px);
        }
        .tIcon {
          width: 34px;
          height: 34px;
          border-radius: 12px;
          display: grid;
          place-items: center;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(15,23,42,0.55);
        }
        .tTitle {
          font-weight: 900;
          font-size: 13px;
          color: rgba(241,245,249,0.92);
        }
        .tSub {
          margin-top: 2px;
          font-size: 12px;
          color: rgba(226,232,240,0.64);
          line-height: 1.35;
        }

        .marquee {
          position: relative;
          margin-top: 16px;
          border-radius: 16px;
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(2,6,23,0.18);
          overflow: hidden;
        }
        .marqueeTrack {
          display: inline-flex;
          gap: 18px;
          padding: 10px 12px;
          white-space: nowrap;
          animation: marquee 18s linear infinite;
          color: rgba(226,232,240,0.62);
          font-weight: 800;
          font-size: 12px;
        }
        .marqueeTrack span {
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(15,23,42,0.35);
        }
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }

        .panelTop {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
          padding: 2px 2px 10px;
        }
        .panelTitle {
          font-weight: 950;
          color: rgba(241,245,249,0.92);
          letter-spacing: -0.02em;
        }
        .panelSub {
          margin-top: 4px;
          font-size: 12.5px;
          color: rgba(226,232,240,0.62);
        }
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-weight: 900;
          font-size: 12px;
          padding: 8px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(2,6,23,0.30);
          color: rgba(241,245,249,0.86);
          white-space: nowrap;
        }

        .panelMedia {
          border-radius: 18px;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(2,6,23,0.30);
          overflow: hidden;
        }
        .panelMediaInner {
          position: relative;
          height: 240px;
        }
        .panelImg {
          object-fit: cover;
          opacity: 0.26;
          transform: scale(1.05);
        }
        .panelMediaOverlay {
          position: absolute;
          inset: 0;
          background:
            radial-gradient(600px 260px at 50% 52%, rgba(59,130,246,0.22), transparent 56%),
            radial-gradient(600px 260px at 80% 20%, rgba(168,85,247,0.16), transparent 56%),
            radial-gradient(600px 260px at 20% 90%, rgba(236,72,153,0.12), transparent 60%),
            linear-gradient(180deg, rgba(2,6,23,0.35), rgba(2,6,23,0.65));
          pointer-events: none;
        }

        .panelRow {
          margin-top: 12px;
          display: grid;
          gap: 10px;
        }

        .pillRow {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .pill {
          font-size: 12px;
          font-weight: 900;
          padding: 7px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(2,6,23,0.22);
          color: rgba(241,245,249,0.86);
        }
        .pill.muted { color: rgba(226,232,240,0.55); }

        .ctxRow {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .ctx {
          font-size: 11.5px;
          font-weight: 850;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(15,23,42,0.45);
          color: rgba(226,232,240,0.72);
        }

        .panelGrid {
          display: grid;
          gap: 12px;
          margin-top: 12px;
        }
        @media (min-width: 980px) {
          .panelGrid { grid-template-columns: 1fr 1fr; }
        }

        .miniCard {
          border-radius: 18px;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(2,6,23,0.22);
          padding: 12px;
        }
        .miniHead {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 10px;
        }
        .miniTitle {
          font-weight: 950;
          color: rgba(241,245,249,0.90);
        }
        .miniTag {
          font-size: 11.5px;
          font-weight: 900;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(15,23,42,0.45);
          color: rgba(226,232,240,0.70);
        }

        .rows {
          display: grid;
          gap: 10px;
        }
        .row {
          display: grid;
          grid-template-columns: 1fr 110px 42px;
          gap: 10px;
          align-items: center;
        }
        .rowName {
          font-weight: 900;
          font-size: 12.5px;
          color: rgba(241,245,249,0.90);
        }
        .rowNote {
          font-size: 12px;
          color: rgba(226,232,240,0.60);
          margin-top: 2px;
          line-height: 1.3;
        }
        .rowBar {
          height: 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(15,23,42,0.55);
          overflow: hidden;
        }
        .rowBar span {
          display: block;
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, rgba(59,130,246,1), rgba(168,85,247,0.95), rgba(236,72,153,0.95));
        }
        .rowScore {
          font-size: 12px;
          font-weight: 950;
          color: rgba(241,245,249,0.84);
          text-align: right;
        }

        .steps {
          display: grid;
          gap: 10px;
          margin-top: 8px;
        }
        .s {
          display: flex;
          gap: 10px;
          align-items: flex-start;
          padding: 10px;
          border-radius: 14px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(15,23,42,0.35);
        }
        .sDot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          margin-top: 4px;
          background: linear-gradient(135deg, rgba(59,130,246,1), rgba(168,85,247,1), rgba(236,72,153,1));
          box-shadow: 0 0 0 6px rgba(59,130,246,0.10);
          flex: 0 0 auto;
        }
        .sT {
          font-weight: 950;
          font-size: 12.5px;
          color: rgba(241,245,249,0.90);
        }
        .sD {
          font-size: 12px;
          color: rgba(226,232,240,0.62);
          margin-top: 2px;
          line-height: 1.35;
        }

        .miniFooter {
          margin-top: 10px;
          display: flex;
          gap: 10px;
          align-items: center;
          font-size: 12px;
          font-weight: 800;
          color: rgba(226,232,240,0.66);
          padding: 10px;
          border-radius: 14px;
          border: 1px solid rgba(148,163,184,0.12);
          background: rgba(2,6,23,0.18);
        }

        .section {
          margin-top: 22px;
        }
        .sectionHead {
          max-width: 760px;
          margin: 0 auto;
          text-align: center;
          padding: 10px 10px 0;
        }
        .eyebrow {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          font-size: 12px;
          font-weight: 900;
          letter-spacing: 0.02em;
          color: rgba(226,232,240,0.62);
          border: 1px solid rgba(148,163,184,0.14);
          background: rgba(2,6,23,0.22);
          padding: 8px 12px;
          border-radius: 999px;
        }
        .h2 {
          margin: 12px 0 0;
          font-size: clamp(26px, 3.2vw, 40px);
          font-weight: 950;
          letter-spacing: -0.03em;
          color: rgba(241,245,249,0.95);
          line-height: 1.08;
        }
        .p {
          margin: 10px auto 0;
          font-size: 14.5px;
          line-height: 1.65;
          color: rgba(226,232,240,0.70);
          max-width: 70ch;
        }

        .cards {
          margin-top: 14px;
          display: grid;
          gap: 12px;
        }
        @media (min-width: 980px) {
          .cards { grid-template-columns: repeat(4, 1fr); }
        }

        .ctaBottom {
          margin-top: 18px;
        }
        .ctaBox {
          border-radius: 22px;
          border: 1px solid rgba(148,163,184,0.16);
          background: linear-gradient(180deg, rgba(15,23,42,0.58), rgba(15,23,42,0.40));
          box-shadow: 0 30px 80px rgba(0,0,0,0.35);
          padding: 16px;
          display: flex;
          flex-wrap: wrap;
          gap: 14px;
          align-items: center;
          justify-content: space-between;
        }
        .ctaTitle {
          font-weight: 950;
          color: rgba(241,245,249,0.94);
          letter-spacing: -0.02em;
          font-size: 16px;
        }
        .ctaSub {
          margin-top: 4px;
          font-size: 13px;
          color: rgba(226,232,240,0.66);
          line-height: 1.45;
        }
        .ctaBtns {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }
      `}</style>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <motion.div
      className="fCard"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.35 }}
      transition={{ duration: 0.5 }}
      whileHover={{ y: -4 }}
    >
      <div className="fIcon">{icon}</div>
      <div className="fTitle">{title}</div>
      <div className="fDesc">{desc}</div>

      <style jsx>{`
        .fCard {
          border-radius: 18px;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(2,6,23,0.22);
          padding: 14px;
          min-height: 150px;
          box-shadow: 0 22px 60px rgba(0,0,0,0.25);
          backdrop-filter: blur(10px);
        }
        .fIcon {
          width: 38px;
          height: 38px;
          border-radius: 14px;
          display: grid;
          place-items: center;
          border: 1px solid rgba(148,163,184,0.16);
          background: rgba(15,23,42,0.50);
          color: rgba(241,245,249,0.88);
        }
        .fTitle {
          margin-top: 10px;
          font-weight: 950;
          color: rgba(241,245,249,0.92);
          letter-spacing: -0.02em;
          font-size: 13.5px;
        }
        .fDesc {
          margin-top: 6px;
          font-size: 12.8px;
          line-height: 1.55;
          color: rgba(226,232,240,0.66);
        }
      `}</style>
    </motion.div>
  );
}

/**
 * Pure SVG “playbook court” overlay — feels like strategy / X&O.
 * No external assets needed.
 */
function PlaybookCourt() {
  return (
    <svg
      className="playbook"
      viewBox="0 0 1200 560"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {/* Court lines */}
      <g opacity="0.72">
        <rect x="70" y="55" width="1060" height="450" rx="24" fill="none" stroke="rgba(226,232,240,0.20)" strokeWidth="2" />
        <line x1="600" y1="55" x2="600" y2="505" stroke="rgba(226,232,240,0.20)" strokeWidth="2" />
        <circle cx="600" cy="280" r="64" fill="none" stroke="rgba(226,232,240,0.18)" strokeWidth="2" />
        <path d="M210,140 h170 v280 h-170" fill="none" stroke="rgba(226,232,240,0.18)" strokeWidth="2" />
        <path d="M990,140 h-170 v280 h170" fill="none" stroke="rgba(226,232,240,0.18)" strokeWidth="2" />
        <path d="M300,280 a220,220 0 0 1 0,-0.01" fill="none" stroke="rgba(226,232,240,0.12)" strokeWidth="2" />
      </g>

      {/* Animated arrows */}
      <g className="arrows" opacity="0.9">
        <path className="dash" d="M360 350 C 460 310, 520 290, 575 250" />
        <path className="dash" d="M650 300 C 740 310, 820 320, 920 260" />
        <path className="solid" d="M540 210 C 610 170, 700 150, 820 165" />
      </g>

      {/* X / O markers */}
      <g opacity="0.95">
        <circle cx="360" cy="350" r="10" fill="rgba(59,130,246,0.95)" />
        <circle cx="650" cy="300" r="10" fill="rgba(168,85,247,0.92)" />
        <circle cx="540" cy="210" r="10" fill="rgba(236,72,153,0.90)" />

        <g stroke="rgba(241,245,249,0.85)" strokeWidth="3" strokeLinecap="round">
          <line x1="445" y1="315" x2="465" y2="335" />
          <line x1="465" y1="315" x2="445" y2="335" />

          <line x1="735" y1="315" x2="755" y2="335" />
          <line x1="755" y1="315" x2="735" y2="335" />
        </g>
      </g>

      <style jsx>{`
        .playbook {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
        }
        .arrows path {
          fill: none;
          stroke-width: 4;
          stroke-linecap: round;
          stroke-linejoin: round;
        }
        .dash {
          stroke: rgba(96,165,250,0.75);
          stroke-dasharray: 10 10;
          animation: dash 1.4s linear infinite;
        }
        .solid {
          stroke: rgba(167,139,250,0.75);
          stroke-dasharray: 520;
          stroke-dashoffset: 520;
          animation: draw 2.2s ease-in-out infinite;
        }

        @keyframes dash {
          to { stroke-dashoffset: -40; }
        }
        @keyframes draw {
          0% { stroke-dashoffset: 520; opacity: 0.55; }
          40% { opacity: 0.95; }
          60% { opacity: 0.95; }
          100% { stroke-dashoffset: 0; opacity: 0.55; }
        }
      `}</style>
    </svg>
  );
}