"use client";

// UI Layer — the signed-in user's saved team, once the page's team list is
// loaded (SCRUM-842). Returns "" until then, or when no valid team is saved.

import { useEffect, useState } from "react";
import { getAccountTeam } from "../../infrastructure/auth";
import { accountTeamFor } from "../../domain/account-team";

export function useAccountTeam(teams: readonly string[] | undefined): string {
  const [saved, setSaved] = useState("");

  useEffect(() => {
    let cancelled = false;
    getAccountTeam()
      .then((team) => {
        if (!cancelled) setSaved(team);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return accountTeamFor(saved, teams ?? []);
}
