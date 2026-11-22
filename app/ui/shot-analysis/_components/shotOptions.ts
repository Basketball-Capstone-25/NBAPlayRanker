import { fetchMetaOptions, fetchPbpMetaOptions } from "../../../services/shotAnalysis";

export type ShotOptions = {
  seasons: string[];
  teams: string[];
  shotTypes: string[];
  zones: string[];
  teamNames: Record<string, string>;
};

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
