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

          <h1 className="auth-title">Welcome back to the film room.</h1>

          <p className="auth-copy">
            Sign in to continue into your role-based workspace and pick up right where your scouting,
            matchup analysis, or evidence review left off.
          </p>

          <div className="auth-point-grid">
            <div className="auth-point">
              <div className="auth-point-title">Coach workflow</div>
              <div className="auth-point-copy">
                Matchup baseline, context-aware recommendations, and a polished gameplan view.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Analyst workflow</div>
              <div className="auth-point-copy">
                Data explorer, metrics, heatmaps, shot analysis, and evidence-first reporting pages.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Explainable output</div>
              <div className="auth-point-copy">
                The product is built to feel like a finished decision-support tool, not a rough demo.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Role-aware access</div>
              <div className="auth-point-copy">
                Navigation stays clean and only surfaces the views that matter to the signed-in user.
              </div>
            </div>
          </div>
        </div>

        <div className="auth-panel">
          <div className="auth-panel-kicker">Log in</div>
          <h2 className="auth-panel-title">Access your workspace</h2>
          <p className="auth-panel-copy">
            Use your account credentials to continue into the platform.
          </p>

          {next ? (
            <div className="auth-inline-note">After login, you will continue to: {next}</div>
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
          <h1 className="auth-title">Welcome back to the film room.</h1>
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
