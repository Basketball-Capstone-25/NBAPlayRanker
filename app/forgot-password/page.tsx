"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { createClient } from "../../lib/supabase/client";

export default function ForgotPasswordPage() {
  const supabase = useMemo(() => createClient(), []);

  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleResetRequest(e: React.FormEvent) {
    e.preventDefault();
    setMessage("");
    setSuccess(false);
    setLoading(true);

    const redirectTo = `${window.location.origin}/auth/callback?next=/reset-password`;

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo,
    });

    if (error) {
      setMessage(error.message);
      setSuccess(false);
      setLoading(false);
      return;
    }

    setMessage(
      "If an account exists for that email, a password reset link has been sent."
    );
    setSuccess(true);
    setLoading(false);
  }

  return (
    <section className="auth-page">
      <div className="auth-shell">
        <div className="auth-stage">
          <div className="auth-badge">NBA Play Ranker • Password recovery</div>

          <h1 className="auth-title">Reset your password.</h1>

          <p className="auth-copy">
            Enter the email tied to your workspace account and we will send you a
            secure reset link.
          </p>

          <div className="auth-point-grid">
            <div className="auth-point">
              <div className="auth-point-title">Secure recovery</div>
              <div className="auth-point-copy">
                Open the link in your email to choose a new password.
              </div>
            </div>

            <div className="auth-point">
              <div className="auth-point-title">Same workspace access</div>
              <div className="auth-point-copy">
                After resetting your password, you can continue into the platform normally.
              </div>
            </div>
          </div>
        </div>

        <div className="auth-panel">
          <div className="auth-panel-kicker">Forgot password</div>
          <h2 className="auth-panel-title">Send reset link</h2>
          <p className="auth-panel-copy">
            We will email you a link to create a new password.
          </p>

          <form className="auth-form" onSubmit={handleResetRequest}>
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

            <button type="submit" className="auth-submit" disabled={loading}>
              {loading ? "Sending..." : "Send reset link"}
            </button>
          </form>

          {message ? (
            <div className={`auth-alert ${success ? "auth-alert-success" : "auth-alert-error"}`}>
              {message}
            </div>
          ) : null}

          <div className="auth-panel-note">
            Remembered it? <Link href="/login">Back to login</Link>
          </div>
        </div>
      </div>
    </section>
  );
}