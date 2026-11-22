export type HeatmapPreset = "ALL" | "RIM" | "CORNER_3" | "ABOVE_BREAK_3" | "MIDRANGE";

const ZONE_ALIASES: Record<Exclude<HeatmapPreset, "ALL">, string[]> = {
  RIM: ["rim", "restrictedarea", "atrim"],
  CORNER_3: ["corner3", "cornerthree"],
  ABOVE_BREAK_3: ["arc3", "abovethebreak3", "abovebreak3"],
  MIDRANGE: ["mid", "midrange"],
};

export function heatmapPresetFilters(preset: HeatmapPreset, zones: string[]) {
  if (preset === "ALL") return { shotType: "", zone: "" };
  const aliases = ZONE_ALIASES[preset];
  const zone = zones.find(value => aliases.includes(value.toLowerCase().replace(/[^a-z0-9]/g, "")));
  // A missing zone must not broaden a specific preset to all shots.
  return zone ? { shotType: "", zone } : null;
}
