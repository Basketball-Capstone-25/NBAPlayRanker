import "./globals.css";
import Nav from "../components/Nav";
import type { ReactNode } from "react";

export const metadata = {
  title: "Basketball Game Strategy Analysis",
  description: "Baseline play-type analysis for NBA matchups.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="container">
          {children}
          <footer className="footer">
            <div>Built with FastAPI &amp; Next.js</div>
          </footer>
        </main>
      </body>
    </html>
  );
}
