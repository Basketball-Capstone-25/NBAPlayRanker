"use client";

import Link from "next/link";
import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn } from "../../../infrastructure/auth";

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const next = searchParams.get("next");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setMessage("");
    setLoading(true);

    try {
      const { role } = await signIn(email, password);
      router.push(next || (role === "coach" ? "/matchup" : "/data-explorer"));
      router.refresh();
    } catch (err: unknown) {
      setMessage(err instanceof Error ? err.message : "Unable to log in right now.");
      setLoading(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-shell">
        <div className="auth-stage">
          <div className="auth-badge">NBA Play Ranker • Secure access</div>

          <h1 className="auth-title">Welcome back.</h1>

          <p className="auth-copy">
            Sign in to compare plays, build a gameplan or explore the data.
          </p>

          <div className="auth-point-grid">
            <div className="auth-point">
              <div className="auth-point-title">Coach workflow</div>
              <div className="auth-point-copy">
                Compare matchups and build a gameplan.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Analyst workflow</div>
              <div className="auth-point-copy">
                Explore the data and check model results.
              </div>
            </div>

          </div>
        </div>

        <div className="auth-panel">
          <div className="auth-panel-kicker">Log in</div>
          <h2 className="auth-panel-title">Access your workspace</h2>
          <p className="auth-panel-copy">
            Enter your email and password.
          </p>

          {next ? (
            <div className="auth-inline-note">Sign in to open the page you selected.</div>
          ) : null}

          {searchParams.get("access") === "pending" ? (
            <div className="auth-inline-note">
              Your account is awaiting role approval. Ask the project administrator to assign coach or analyst access, then sign in again.
            </div>
          ) : null}

          <form className="auth-form" onSubmit={handleLogin}>
            <label className="auth-field">
              <span className="auth-label">Email</span>
              <input
                className="auth-input"
                type="email"
                placeholder="coach@team.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                required
              />
            </label>

            <label className="auth-field">
              <span className="auth-label">Password</span>

              <div className="auth-input-wrap">
                <input
                  className="auth-input"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />

                <button
                  type="button"
                  className="auth-toggle"
                  onClick={() => setShowPassword((v) => !v)}
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
            </label>

            <button type="submit" className="auth-submit" disabled={loading}>
              {loading ? "Signing in..." : "Log in"}
            </button>
          </form>

          {message ? <div className="auth-alert auth-alert-error">{message}</div> : null}

          <div className="auth-panel-note">
            Need an account? <Link href="/signup">Create one here</Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function LoginPageFallback() {
  return (
    <section className="auth-page">
      <div className="auth-shell">
        <div className="auth-stage">
          <div className="auth-badge">NBA Play Ranker • Secure access</div>
          <h1 className="auth-title">Welcome back.</h1>
          <p className="auth-copy">Loading login...</p>
        </div>

        <div className="auth-panel">
          <div className="auth-panel-kicker">Log in</div>
          <h2 className="auth-panel-title">Access your workspace</h2>
          <p className="auth-panel-copy">Loading login form...</p>
        </div>
      </div>
    </section>
  );
}

export default function LoginPageClient() {
  return (
    <Suspense fallback={<LoginPageFallback />}>
      <LoginPageContent />
    </Suspense>
  );
}
