// Domain layer — the team a user saved on their account (SCRUM-842).
// It is the only input a page may pre-fill, and only when it is a team that
// exists in the loaded data. Anything else is ignored rather than guessed.

/** Read the saved team from Supabase user metadata. */
export function teamFromMetadata(metadata: unknown): string {
  if (!metadata || typeof metadata !== "object") return "";
  const team = (metadata as { team?: unknown }).team;
  return typeof team === "string" ? team.trim().toUpperCase() : "";
}

/** The account team to pre-fill, or "" when it is not one of the loaded teams. */
export function accountTeamFor(saved: string, teams: readonly string[]): string {
  return saved && teams.includes(saved) ? saved : "";
}
