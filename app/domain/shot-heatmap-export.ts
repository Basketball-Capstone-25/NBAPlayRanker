export type ShotHeatmapExportInputs = {
  season: string;
  team: string;
  opp: string;
  shotType: string;
  zone: string;
  maxShots: number;
  version: string;
};

const TEAMS = new Set("ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split(" "));
const KEYS = new Set(["season", "team", "opp", "shot_type", "zone", "max_shots", "version"]);

export function parseShotHeatmapExport(params: URLSearchParams): ShotHeatmapExportInputs | null {
  for (const key of params.keys()) if (!KEYS.has(key) || params.getAll(key).length !== 1) return null;
  const season = params.get("season") ?? "";
  const team = params.get("team") ?? "";
  const opp = params.get("opp") ?? "";
  const shotType = params.get("shot_type") ?? "";
  const zone = params.get("zone") ?? "";
  const cap = params.get("max_shots") ?? "";
  const version = params.get("version") ?? "";
  const maxShots = Number(cap);
  if (!/^\d{4}-\d{2}$/.test(season) || Number(season.slice(-2)) !== (Number(season.slice(0, 4)) + 1) % 100 ||
      !TEAMS.has(team) || !TEAMS.has(opp) || team === opp || !/^\d+$/.test(cap) ||
      !Number.isSafeInteger(maxShots) || maxShots < 1000 || maxShots > 100000 ||
      !/^sha256:[a-f0-9]{64}$/.test(version) ||
      [shotType, zone].some(value => value.length > 80 || /[\u0000-\u001f\u007f]/.test(value))) return null;
  return { season, team, opp, shotType, zone, maxShots, version };
}

export function shotHeatmapExportUrl(inputs: Omit<ShotHeatmapExportInputs, "version">, version: unknown): string | null {
  if (typeof version !== "string") return null;
  const params = new URLSearchParams({ season: inputs.season, team: inputs.team, opp: inputs.opp,
    max_shots: String(inputs.maxShots), version });
  if (inputs.shotType) params.set("shot_type", inputs.shotType);
  if (inputs.zone) params.set("zone", inputs.zone);
  return parseShotHeatmapExport(params) ? `/api/exports/shot-heatmap?${params}` : null;
}

export function shotHeatmapFilename(inputs: ShotHeatmapExportInputs): string {
  const safe = (value: string) => value.replace(/[^a-z0-9_-]+/gi, "_");
  return `shot_heatmap_${inputs.season}_${inputs.team}_vs_${inputs.opp}_${safe(inputs.shotType || "alltypes")}_${safe(inputs.zone || "allzones")}_max_${inputs.maxShots}.png`;
}
