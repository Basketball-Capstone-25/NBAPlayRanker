export type DatasetMetadata = {
  id: string; name: string; file: string; source: string; status: string;
  version: string | null; pulled_at: string | null; seasons: string[];
  row_count: number | null; columns: string[]; warnings?: string[]; error?: string;
};
export type DictionaryColumn = {
  dataset: string; dataset_name: string; column: string; meaning: string;
  unit: string; source: string; documented: boolean;
};

export function parseDatasetCatalog(value: unknown): DatasetMetadata[] {
  const data = value as { datasets?: unknown };
  if (!data || !Array.isArray(data.datasets)) throw new Error("Dataset metadata unavailable.");
  return data.datasets.map((item: unknown) => {
    const d = item as DatasetMetadata;
    if (!d || typeof d.id !== "string" || typeof d.name !== "string" ||
        !Array.isArray(d.seasons) || !Array.isArray(d.columns) ||
        !(d.row_count === null || Number.isInteger(d.row_count) && d.row_count >= 0) ||
        !(d.version === null || /^sha256:[a-f0-9]{64}$/.test(d.version))) {
      throw new Error("Dataset metadata unavailable: invalid response.");
    }
    return d;
  });
}
