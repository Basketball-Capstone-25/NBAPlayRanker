// app/layout.tsx
//
// Root layout + global navigation.
//
// Updated goals:
// - Stronger branding ("NBA Play Ranker")
// - Dynamic navbar (active link highlight + mobile menu)
// - Keep reviewer-friendly flow + clean footer

import "./globals.css";
import type { ReactNode } from "react";
import Link from "next/link";
import NavBar from "./ui/_shared/components/NavBar";


export const metadata = {
  title: "NBA Play Ranker",
  description:
    "A decision-support web app that ranks offensive play types using an explainable baseline and an AI context simulator, backed by model evaluation and statistical analysis.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <NavBar />

        <main className="container">
          {children}

          <footer className="footer">
            <div>
              Built with <strong>Next.js</strong> (UI) + <strong>FastAPI</strong> (services).
            </div>
            <div>
              Tip for reviewers: start at <Link href="/data-explorer">Data Explorer</Link> to see the dataset,
              then compare <Link href="/matchup">Baseline</Link> vs <Link href="/context">AI Context</Link>,
              and verify evidence in <Link href="/model-metrics">Model Performance</Link> and{" "}
              <Link href="/statistical-analysis">Statistical Analysis</Link>.
            </div>
          </footer>
        </main>
      </body>
    </html>
  );
}
