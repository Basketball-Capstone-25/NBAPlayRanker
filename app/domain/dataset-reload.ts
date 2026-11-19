export type ReloadDataset = "synergy" | "pbp";

export type DatasetReloadResult = {
  dataset: "synergy" | "pbp_shots";
  status: "reloaded";
  source_file: string;
  previous_version: string;
  dataset_version: string;
  changed: boolean;
  rows: number;
  seasons: string[];
  quality_status?: "pass";
};

export class DatasetReloadError extends Error {
  constructor(message: string, public readonly outcome: "not_started" | "rejected" | "unknown", public readonly status?: number) {
    super(message);
    this.name = "DatasetReloadError";
  }
}

const text = (value: unknown, max: number): value is string =>
  typeof value === "string" && value.trim().length > 0 && value.length <= max && !/[\u0000-\u001f\u007f]/.test(value);

/** A successful POST must identify the requested dataset and its loaded version. */
export function parseDatasetReload(value: unknown, expected: ReloadDataset): DatasetReloadResult {
  const invalid = (): never => {
    throw new DatasetReloadError("The reload reply was incomplete. Check the data before trying again.", "unknown");
  };
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  const reply = value as Record<string, unknown>;
  const dataset = expected === "synergy" ? "synergy" : expected === "pbp" ? "pbp_shots" : null;
  if (!dataset || reply.dataset !== dataset || reply.status !== "reloaded"
    || !text(reply.source_file, 255) || /[/\\]/.test(reply.source_file)
    || !text(reply.previous_version, 160) || !text(reply.dataset_version, 160)
    || typeof reply.changed !== "boolean" || reply.changed !== (reply.previous_version !== reply.dataset_version)
    || typeof reply.rows !== "number" || !Number.isSafeInteger(reply.rows) || reply.rows <= 0
    || !Array.isArray(reply.seasons) || reply.seasons.length === 0 || reply.seasons.length > 100
    || new Set(reply.seasons).size !== reply.seasons.length
    || !reply.seasons.every(season => typeof season === "string" && /^\d{4}-\d{2}$/.test(season))
    || (dataset === "synergy" && reply.quality_status !== "pass")) return invalid();
  return {
    dataset, status: "reloaded", source_file: reply.source_file,
    previous_version: reply.previous_version, dataset_version: reply.dataset_version,
    changed: reply.changed, rows: reply.rows, seasons: [...reply.seasons],
    ...(dataset === "synergy" ? { quality_status: "pass" as const } : {}),
  };
}
