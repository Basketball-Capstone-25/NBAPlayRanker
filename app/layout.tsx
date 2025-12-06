import "./globals.css";
import Nav from "../components/Nav";
import type { ReactNode } from "react";

export const metadata = {
  title: "Basketball Game Strategy Analysis",
  description: "PSPI – baseline play-type recommender demo (FastAPI + Next.js)",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="container">
          {children}
          <footer className="footer">
            <div>Basketball Game Strategy Analysis</div>
            <div>Frontend: Next.js App Router • Backend: FastAPI baseline API.</div>
          </footer>
        </main>
      </body>
    </html>
  );
}
