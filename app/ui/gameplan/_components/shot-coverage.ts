// Play-type seasons and recorded-shot seasons are separate data sources.
export async function loadShotsForSeason<T>(season: string, knownShotSeasons: string[] | null, load: () => Promise<T>): Promise<T | null> {
  if (knownShotSeasons !== null && !knownShotSeasons.includes(season)) return null;
  return load();
}
