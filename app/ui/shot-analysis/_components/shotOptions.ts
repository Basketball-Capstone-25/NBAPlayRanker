// Dropdown options shared by the three shot search pages.

import { fetchMetaOptions, fetchPbpMetaOptions } from "../../../services/shotAnalysis";

export type ShotOptions = {
  seasons: string[];
  teams: string[];
  shotTypes: string[];
  zones: string[];
  teamNames: Record<string, string>;
};

/** Gets valid seasons, teams, shot types and zones from play-by-play. The play-type source supplies optional team labels only. */
export async function loadShotOptions(): Promise<ShotOptions> {
  const [shots, labels] = await Promise.all([
    fetchPbpMetaOptions({ strict: true }),
    fetchMetaOptions().catch(() => null),
  ]);
  if (!shots.seasons.length || !shots.teams.length) {
    throw new Error("No shot seasons or teams are available. Try loading the filters again.");
  }
  return {
    seasons: shots.seasons,
    teams: shots.teams,
    shotTypes: shots.shotTypes ?? [],
    zones: shots.zones ?? [],
    teamNames: labels?.teamNames ?? {},
  };
}
