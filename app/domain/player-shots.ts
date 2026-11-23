// app/domain/player-shots.ts
//
// Domain Layer — a player's made shots on a half court (SCRUM-825).
// Points come from the API in feet: x from midcourt (0) to the baseline (47),
// the basket at x = 41.75, y across the court from -25 to 25. The court is
// drawn with the basket at the top, 10 SVG units per foot.

export const COURT = { width: 500, height: 470, scale: 10, basketX: 41.75 };
export const CELL_FEET = 2;

export const toSvgX = (yFeet: number) => (yFeet + 25) * COURT.scale;
export const toSvgY = (xFeet: number) => (47 - xFeet) * COURT.scale;

export type HeatCell = { col: number; row: number; count: number; share: number };

/** Count made shots in 2 ft squares; share = count / the busiest square (0–1). */
export function heatCells(points: readonly (readonly [number, number])[], cell = CELL_FEET): HeatCell[] {
  const counts = new Map<string, { col: number; row: number; count: number }>();
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 47 || Math.abs(y) > 25) continue;
    const col = Math.min(Math.floor((y + 25) / cell), Math.floor(50 / cell) - 1);
    const row = Math.min(Math.floor((47 - x) / cell), Math.ceil(47 / cell) - 1);
    const key = `${col}:${row}`;
    const found = counts.get(key);
    if (found) found.count += 1;
    else counts.set(key, { col, row, count: 1 });
  }
  const max = Math.max(0, ...[...counts.values()].map((c) => c.count));
  return [...counts.values()]
    .sort((a, b) => a.row - b.row || a.col - b.col)
    .map((c) => ({ ...c, share: max ? c.count / max : 0 }));
}

/** Fill opacity for a square: busier squares are a deeper green, never invisible. */
export const cellOpacity = (share: number) => 0.2 + 0.75 * Math.max(0, Math.min(1, share));

/** Seasons as readable text, e.g. "2023-24 and 2024-25". */
export function listSeasons(seasons: readonly string[]): string {
  if (seasons.length <= 2) return seasons.join(" and ");
  return `${seasons.slice(0, -1).join(", ")} and ${seasons[seasons.length - 1]}`;
}

/** One sentence describing the made-shot map, used as its caption and alt text. */
export function describeMadeShots(r: {
  player: { name: string }; opp: string; seasons: string[]; made: number; shown: number; player_games: number;
  excluded_location: number; outside_halfcourt: number;
}): string {
  const when = `against ${r.opp} in ${listSeasons(r.seasons)}`;
  if (r.made === 0) return `${r.player.name} has no made shots ${when}.`;
  const games = `${r.player_games} game${r.player_games === 1 ? "" : "s"}`;
  let text = `${r.player.name}: ${r.made} made shot${r.made === 1 ? "" : "s"} in ${games} ${when}.`;
  const hidden = r.made - r.shown;
  if (hidden > 0) text += ` ${hidden} without a usable court location ${hidden === 1 ? "is" : "are"} not shown.`;
  return text;
}
