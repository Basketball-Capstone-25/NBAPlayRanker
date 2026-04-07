"use client";

import Link from "next/link";
import { useState } from "react";
import { signUp, type UserRole } from "../../../infrastructure/auth";

export default function SignupPageClient() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("coach");
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    setMessage("");
    setSuccess(false);
    setLoading(true);

    try {
      await signUp(email, password, role);
      setMessage("Check your email to verify your account, then sign in.");
      setSuccess(true);
    } catch (err: unknown) {
      setMessage(err instanceof Error ? err.message : "Sign-up failed.");
      setSuccess(false);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-shell">
        <div className="auth-stage">
          <div className="auth-badge">NBA Play Ranker • Account setup</div>

          <h1 className="auth-title">Create your role-based account.</h1>

          <p className="auth-copy">
            Pick the workspace that matches how you use the platform. Coaches get decision-ready
            recommendations. Analysts get the deeper evidence, metrics, and exploratory views.
          </p>

          <div className="auth-point-grid">
            <div className="auth-point">
              <div className="auth-point-title">Coach access</div>
              <div className="auth-point-copy">
                Built for fast possession decisions, context-aware ranking, and polished game planning.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Analyst access</div>
              <div className="auth-point-copy">
                Built for exploration, traceability, evaluation, heatmaps, and deeper statistical views.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Same product language</div>
              <div className="auth-point-copy">
                Everything stays on-theme with explainability, evidence, and basketball workflow in mind.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Clean navigation</div>
              <div className="auth-point-copy">
                Once signed in, the navbar expands into the correct workspace instead of looking cluttered.
              </div>
            </div>
          </div>
        </div>

        <div className="auth-panel">
          <div className="auth-panel-kicker">Sign up</div>
          <h2 className="auth-panel-title">Create your account</h2>
          <p className="auth-panel-copy">
            Choose a role, then set your credentials to get started.
          </p>

          <form className="auth-form" onSubmit={handleSignup}>
            <div className="auth-field">
              <span className="auth-label">Choose your role</span>

              <div className="auth-role-grid">
                <button
                  type="button"
                  className={`auth-role-card ${role === "coach" ? "active" : ""}`}
                  onClick={() => setRole("coach")}
                  aria-pressed={role === "coach"}
                >
                  <div className="auth-role-title">Coach</div>
                  <div className="auth-role-copy">
                    Matchup, context recommendations, and gameplan workflow.
                  </div>
                </button>

                <button
                  type="button"
                  className={`auth-role-card ${role === "analyst" ? "active" : ""}`}
                  onClick={() => setRole("analyst")}
                  aria-pressed={role === "analyst"}
                >
                  <div className="auth-role-title">Analyst</div>
                  <div className="auth-role-copy">
                    Data explorer, model metrics, shot views, and statistical analysis.
                  </div>
                </button>
              </div>
            </div>

            <label className="auth-field">
              <span className="auth-label">Email</span>
              <input
                className="auth-input"
                type="email"
                placeholder="analyst@team.com"
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
                  placeholder="At least 8 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
                  minLength={8}
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
              {loading ? "Creating account..." : "Create account"}
            </button>
          </form>

          {message ? (
            <div className={`auth-alert ${success ? "auth-alert-success" : "auth-alert-error"}`}>
              {message}
            </div>
          ) : null}

          <div className="auth-panel-note">
            Already have an account? <Link href="/login">Log in</Link>
          </div>
        </div>
      </div>
    </section>
  );
}
