"use client";

// UI Layer — account settings (SCRUM-842). The only setting is the user's team,
// which pages may pre-fill. Teams come from the loaded data, never a fixed list.

import React, { useEffect, useState } from "react";
import { getAccountTeam, setAccountTeam } from "../../../infrastructure/auth";
import { fetchMetaOptions } from "../../../infrastructure/api-client";

export default function AccountPageClient() {
  const [teams, setTeams] = useState<string[]>([]);
  const [team, setTeam] = useState("");
  const [saved, setSaved] = useState("");
  const [status, setStatus] = useState<"loading" | "ready" | "saving">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [meta, current] = await Promise.all([fetchMetaOptions(), getAccountTeam().catch(() => "")]);
      if (cancelled) return;
      setTeams(meta.teams ?? []);
      setTeam(current);
      setSaved(current);
      setStatus("ready");
      if (meta._fallback) setMessage("Teams are unavailable right now. Try again later.");
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    setStatus("saving");
    setMessage("");
    try {
      await setAccountTeam(team);
      setSaved(team);
      setMessage(team ? `Saved. Pages will start with ${team} as your team.` : "Cleared. Pages will start with no team chosen.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Could not save your team. Try again.");
    } finally {
      setStatus("ready");
    }
  }

  return (
    <main className="container" style={{ maxWidth: 560 }}>
      <h1 className="h1">Account</h1>
      <form className="card" onSubmit={save} style={{ display: "grid", gap: 12 }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span>My team (optional)</span>
          <select className="input" value={team} onChange={(e) => setTeam(e.target.value)} disabled={status !== "ready" || teams.length === 0}>
            <option value="">No team</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          Pages pre-fill this as &quot;our team&quot;. Nothing else is filled in from your account.
        </p>
        <button className="btn btnPrimary" type="submit" disabled={status !== "ready" || team === saved}>
          {status === "saving" ? "Saving…" : "Save"}
        </button>
        {message && <p role="status" style={{ margin: 0 }}>{message}</p>}
      </form>
    </main>
  );
}
