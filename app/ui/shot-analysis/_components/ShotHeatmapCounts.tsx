import React from "react";
import type { ShotHeatmapResponse } from "../../../services/shotAnalysis";

export function ShotHeatmapCounts({ heatmap }: { heatmap: ShotHeatmapResponse }) {
  const shown = heatmap.n_shots_rendered ?? heatmap.n_shots;
  const count = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const parts: string[] = [];
  if (count(heatmap.n_shots_total)) parts.push(`Matching attempts: ${heatmap.n_shots_total!.toLocaleString()}`);
  if (count(shown)) parts.push(`Attempts shown: ${shown!.toLocaleString()}`);
  if (heatmap.n_shots_excluded_location > 0) parts.push(`Unusable locations excluded: ${heatmap.n_shots_excluded_location.toLocaleString()}`);
  if (heatmap.n_shots_outside_halfcourt > 0) parts.push(`Outside the half court: ${heatmap.n_shots_outside_halfcourt.toLocaleString()}`);
  if (heatmap.n_shots_omitted_by_cap > 0) parts.push(`Not shown due to sample cap: ${heatmap.n_shots_omitted_by_cap.toLocaleString()}`);
  return parts.length ? <p className="muted">{parts.join(" • ")}</p> : null;
}
