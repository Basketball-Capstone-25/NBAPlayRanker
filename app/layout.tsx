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
            <div>NBA Play Ranker · Basketball decision support</div>
            <div><Link href="/glossary">Glossary</Link> · <Link href="/data-dictionary">Data Dictionary</Link></div>
          </footer>
        </main>
      </body>
    </html>
  );
}
